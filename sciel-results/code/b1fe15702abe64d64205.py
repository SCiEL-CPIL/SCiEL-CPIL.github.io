import time

DIMENSIONS = [3, 4, 6]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _strassen_2x2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur):
    def u(i, j): return (i_off + i) * m + (j_off + j)
    def v(j, k): return (j_off + j) * p + (k_off + k)
    def w(i, k): return (k_off + k) * n + (i_off + i)

    # M1 = (a00+a11)(b00+b11)
    U[u(0,0)][cur] += 1
    U[u(1,1)][cur] += 1
    V[v(0,0)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M2 = (a10+a11)(b00)
    U[u(1,0)][cur] += 1
    U[u(1,1)][cur] += 1
    V[v(0,0)][cur] += 1
    W[w(1,0)][cur] += 1
    W[w(1,1)][cur] -= 1
    cur += 1
    # M3 = (a00)(b01-b11)
    U[u(0,0)][cur] += 1
    V[v(0,1)][cur] += 1
    V[v(1,1)][cur] -= 1
    W[w(0,1)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M4 = (a11)(b10-b00)
    U[u(1,1)][cur] += 1
    V[v(1,0)][cur] += 1
    V[v(0,0)][cur] -= 1
    W[w(0,0)][cur] += 1
    W[w(1,0)][cur] += 1
    cur += 1
    # M5 = (a00+a01)(b11)
    U[u(0,0)][cur] += 1
    U[u(0,1)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] -= 1
    W[w(0,1)][cur] += 1
    cur += 1
    # M6 = (a10-a00)(b00+b01)
    U[u(1,0)][cur] += 1
    U[u(0,0)][cur] -= 1
    V[v(0,0)][cur] += 1
    V[v(0,1)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1
    # M7 = (a01-a11)(b10+b11)
    U[u(0,1)][cur] += 1
    U[u(1,1)][cur] -= 1
    V[v(1,0)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,0)][cur] += 1
    cur += 1

    return cur

def _winograd_2x2_2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur):
    def u(i, j): return (i_off + i) * m + (j_off + j)
    def v(j, k): return (j_off + j) * p + (k_off + k)
    def w(i, k): return (k_off + k) * n + (i_off + i)

    # Winograd F(2x2, 2x2): 8 multiplications
    # C = [a00 a01; a10 a11] * [b00 b01; b10 b11]
    # m1 = (a00 + a01)(b00 + b10)
    U[u(0,0)][cur] += 1
    U[u(0,1)][cur] += 1
    V[v(0,0)][cur] += 1
    V[v(0,1)][cur] += 1
    W[w(0,0)][cur] += 1
    W[w(1,0)][cur] += 1
    cur += 1
    # m2 = (a10 - a00)(b00 + b01)
    U[u(1,0)][cur] += 1
    U[u(0,0)][cur] -= 1
    V[v(0,0)][cur] += 1
    V[v(1,0)][cur] += 1
    W[w(0,0)][cur] -= 1
    W[w(0,1)][cur] += 1
    cur += 1
    # m3 = (a10 - a11)(b11)
    U[u(1,0)][cur] += 1
    U[u(1,1)][cur] -= 1
    V[v(1,1)][cur] += 1
    W[w(1,0)][cur] += 1
    W[w(1,1)][cur] -= 1
    cur += 1
    # m4 = (a10 + a11)(b01)
    U[u(1,0)][cur] += 1
    U[u(1,1)][cur] += 1
    V[v(0,1)][cur] += 1
    W[w(0,0)][cur] += 1
    W[w(0,1)][cur] += 1
    cur += 1
    # m5 = a00*b00
    U[u(0,0)][cur] += 1
    V[v(0,0)][cur] += 1
    W[w(0,0)][cur] += 1
    cur += 1
    # m6 = a00*b10
    U[u(0,0)][cur] += 1
    V[v(1,0)][cur] += 1
    W[w(1,0)][cur] += 1
    cur += 1
    # m7 = a01*b11
    U[u(0,1)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(0,1)][cur] += 1
    cur += 1
    # m8 = a11*b11
    U[u(1,1)][cur] += 1
    V[v(1,1)][cur] += 1
    W[w(1,1)][cur] += 1
    cur += 1

    return cur

def _schoolbook(n_block, m_block, p_block, i_off, j_off, k_off, m, n, p, U, V, W, cur):
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                u_idx = (i_off + i) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + (i_off + i)
                U[u_idx][cur] = 1
                V[v_idx][cur] = 1
                W[w_idx][cur] = 1
                cur += 1
    return cur

def _decompose(n, m, p):
    # Returns list of (n_sub, m_sub, p_sub, i_off, j_off, k_off, method)
    # where method is 'strassen', 'winograd', 'schoolbook'
    blocks = []
    
    # Strategy: split n=3 into 1+2, use Winograd for 2x2x2 blocks
    # For n=1: schoolbook on 1xm x mxp
    # For n=2: split m into chunks of 2, use Winograd for 2x2x2
    # But p=6 is even, so we can also split p
    
    # Let's try: n=3 = 1 + 2
    # n=1 part: 1x4x6 -> schoolbook = 24
    # n=2 part: 2x4x6 -> split m=4 into 2+2
    #   2x2x6 -> split p=6 into 2+2+2
    #   3 blocks of 2x2x2, each Winograd = 8, total = 24
    # Total: 24 + 24 = 48
    
    # Alternative: use Strassen for some 2x2x2 blocks (7 vs 8)
    # If we use Strassen: 24 + 3*7 = 24 + 21 = 45
    
    # Let's go with the 45-rank solution using Strassen
    # But first, let me verify the structure is correct
    
    # n=1 block: i_off=0
    blocks.append((1, 4, 6, 0, 0, 0, 'schoolbook'))
    
    # n=2 blocks: i_off=1, split m=4 into 2+2
    for m_chunk in range(2):
        j_off = m_chunk * 2
        # 2x2x6 -> split p=6 into 2+2+2
        for p_chunk in range(3):
            k_off = p_chunk * 2
            blocks.append((2, 2, 2, 1, j_off, k_off, 'strassen'))
    
    return blocks

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    # Try to find a better decomposition using search
    # For now, use the structured approach
    
    blocks = _decompose(n, m, p)
    
    # Calculate total rank
    total_rank = 0
    for n_sub, m_sub, p_sub, i_off, j_off, k_off, method in blocks:
        if method == 'schoolbook':
            total_rank += n_sub * m_sub * p_sub
        elif method == 'strassen':
            total_rank += 7
        elif method == 'winograd':
            total_rank += 8
    
    # Allocate matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    cur = 0
    for n_sub, m_sub, p_sub, i_off, j_off, k_off, method in blocks:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        if method == 'schoolbook':
            cur = _schoolbook(n_sub, m_sub, p_sub, i_off, j_off, k_off, m, n, p, U, V, W, cur)
        elif method == 'strassen':
            cur = _strassen_2x2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur)
        elif method == 'winograd':
            cur = _winograd_2x2_2x2(i_off, j_off, k_off, m, n, p, U, V, W, cur)
    
    return {"U": U, "V": V, "W": W}
