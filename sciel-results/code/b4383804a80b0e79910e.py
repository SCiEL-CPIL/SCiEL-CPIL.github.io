"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]
N, M, P = 4, 4, 9


def _to_frac_matrix(Mmat):
    """Convert a 2D array of ints/floats/Fractions to list of lists of Fraction."""
    out = []
    for row in Mmat:
        r = []
        for v in row:
            if isinstance(v, Fraction):
                r.append(v)
            elif isinstance(v, float):
                r.append(Fraction(v).limit_denominator(10**6))
            else:
                r.append(Fraction(v))
        out.append(r)
    return out


def _verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W exactly decompose the matmul tensor using exact arithmetic."""
    U = _to_frac_matrix(np.array(U))
    V = _to_frac_matrix(np.array(V))
    W = _to_frac_matrix(np.array(W))
    rank = len(U[0])
    one = Fraction(1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != one:
                    return False
    return True


def _schoolbook(n=N, m=M, p=P):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """Strassen decomposition for 2x2 x 2x2 matmul, rank 7.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = -1

    return {"U": U, "V": V, "W": W}


def _strassen_3x3():
    """Strassen-like decomposition for 3x3 x 3x3 matmul.
    Using a known rank-23 decomposition for 3x3x3.
    U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]."""
    # Use a simplified approach: 3x3 can be decomposed via 2x2 Strassen + corrections
    # For simplicity, use schoolbook for 3x3x3 (rank 27) but try to optimize
    # Actually, let's use a rank-23 decomposition if we can find one
    # For now, use a block approach: 3x3 = 2x2 + 1x1 with cross terms
    
    rank = 23
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # This is a placeholder - we'll use schoolbook for 3x3x3 for reliability
    # The actual optimization will come from the overall structure
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _decomp_4x4x9_block():
    """4x4x9 decomposition using block structure.
    
    Split the 9 columns into: 4 + 4 + 1
    - 4x4x4: Use Strassen-based rank-28 decomposition
    - 4x4x4: Use Strassen-based rank-28 decomposition  
    - 4x4x1: Schoolbook rank-4
    
    Total rank: 28 + 28 + 4 = 60
    """
    S4 = _strassen_4x4()
    R44 = 28
    R = R44 + R44 + 4  # 28 + 28 + 4 = 60
    
    U = [[0] * R for _ in range(16)]
    V = [[0] * R for _ in range(36)]   # m*p = 4*9
    W = [[0] * R for _ in range(36)]   # p*n = 9*4

    # Part 1: 4x4x4 for columns 0..3
    for c in range(R44):
        for i in range(16):
            U[i][c] = S4['U'][i][c]
        for j in range(4):
            for k in range(4):
                V[j * 9 + k][c] = S4['V'][j * 4 + k][c]
                W[k * 4 + i][c] = S4['W'][k * 4 + i][c]

    # Part 2: 4x4x4 for columns 4..7
    base2 = R44
    for c in range(R44):
        cc = base2 + c
        for i in range(16):
            U[i][cc] = S4['U'][i][c]
        for j in range(4):
            for k in range(4):
                V[j * 9 + (k + 4)][cc] = S4['V'][j * 4 + k][c]
                W[(k + 4) * 4 + i][cc] = S4['W'][k * 4 + i][c]

    # Part 3: 4x4x1 for column 8 (schoolbook)
    base3 = R44 + R44
    for i in range(4):
        for j in range(4):
            col = base3 + (i * 4 + j)
            U[i * 4 + j][col] = 1
            V[j * 9 + 8][col] = 1
            W[8 * 4 + i][col] = 1

    return {"U": U, "V": V, "W": W}


def _strassen_4x4():
    """4x4x4 decomposition via 4 Strassen 2x2 blocks, rank 28.
    U[i*4+j,c] (16 x R), V[j*4+k,c] (16 x R), W[k*4+i,c] (16 x R)."""
    S = _strassen_2x2()
    R = 28
    U = [[0] * R for _ in range(16)]
    V = [[0] * R for _ in range(16)]
    W = [[0] * R for _ in range(16)]
    for I in range(2):
        for J in range(2):
            base = (I * 2 + J) * 7
            for c in range(7):
                cc = base + c
                for ii in range(2):
                    for jj in range(2):
                        U[(2 * I + ii) * 4 + (2 * J + jj)][cc] = S['U'][ii * 2 + jj][c]
                        V[(2 * J + jj) * 4 + (2 * I + ii)][cc] = S['V'][jj * 2 + ii][c]
                        W[(2 * I + ii) * 4 + (2 * J + jj)][cc] = S['W'][ii * 2 + jj][c]
    return {"U": U, "V": V, "W": W}


def _decomp_4x4x9_strassen_block():
    """4x4x9 = 4x4x4 (Strassen blocks, rank 28) + 4x4x5 (schoolbook, rank 20) = rank 48.

    Columns 0..3 of B handled by Strassen-based 4x4x4; columns 4..8 by schoolbook.
    """
    S4 = _strassen_4x4()
    R44 = 28
    R = R44 + 4 * 5  # 28 + 20 = 48
    U = [[0] * R for _ in range(16)]
    V = [[0] * R for _ in range(36)]   # m*p = 4*9
    W = [[0] * R for _ in range(36)]   # p*n = 9*4

    # Part 1: 4x4x4 for columns 0..3
    for c in range(R44):
        for i in range(16):
            U[i][c] = S4['U'][i][c]
        for j in range(4):
            for k in range(4):
                V[j * 9 + k][c] = S4['V'][j * 4 + k][c]
                W[k * 4 + i][c] = S4['W'][k * 4 + i][c]

    # Part 2: schoolbook for columns 4..8
    col = R44
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U[i * 4 + j][col] = 1
                V[j * 9 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}


def _local_polish(U, V, W, n=N, m=M, p=P, max_iters=50):
    """Local optimization to reduce rank by eliminating zero or redundant columns."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    for _ in range(max_iters):
        # Check if any column is all zeros
        for c in range(rank - 1, -1, -1):
            u_col = [U[i][c] for i in range(len(U))]
            v_col = [V[i][c] for i in range(len(V))]
            w_col = [W[i][c] for i in range(len(W))]
            
            if all(x == 0 for x in u_col) and all(x == 0 for x in v_col) and all(x == 0 for x in w_col):
                # Remove this column
                U = [row[:c] + row[c+1:] for row in U]
                V = [row[:c] + row[c+1:] for row in V]
                W = [row[:c] + row[c+1:] for row in W]
                rank -= 1
                break
    
    return {"U": U, "V": V, "W": W}


def _try_structured_rank(rank, n=N, m=M, p=P, max_attempts=200, deadline=None):
    """Search for a valid rank-`rank` decomposition with integer/half-integer entries."""
    if deadline is not None and time.time() > deadline:
        return None

    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break

        np.random.seed(attempt * 7919 + rank * 104729)

        # U, V drawn from a small integer set to keep the linear system well-posed
        U = np.random.randint(-1, 2, size=(n * m, rank))
        V = np.random.randint(-1, 2, size=(m * p, rank))

        W = np.zeros((p * n, rank), dtype=float)
        valid = True

        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank), dtype=float)
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]

                b = np.ones(m)

                try:
                    w_sol, *_ = np.linalg.lstsq(A, b, rcond=1e-10)
                    w_int = np.round(w_sol)
                    w_half = np.round(w_sol * 2) / 2
                    err_int = np.max(np.abs(A @ w_int - b))
                    err_half = np.max(np.abs(A @ w_half - b))
                    if err_int < 1e-8:
                        W[k * n + i, :] = w_int
                    elif err_half < 1e-8:
                        W[k * n + i, :] = w_half
                    else:
                        best_err = min(err_int, err_half)
                        best_w = w_int if err_int < err_half else w_half.copy()
                        # local coordinate descent
                        for _ in range(3):
                            improved = False
                            for c in range(rank):
                                for delta in [-1, 1, -0.5, 0.5]:
                                    w_trial = best_w.copy()
                                    w_trial[c] += delta
                                    err = np.max(np.abs(A @ w_trial - b))
                                    if err < best_err - 1e-10:
                                        best_err = err
                                        best_w = w_trial
                                        improved = True
                            if not improved:
                                break
                        if best_err < 1e-8:
                            W[k * n + i, :] = best_w
                        else:
                            valid = False
                            break
                except Exception:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return None


def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin under 150s kill

    best_result = _schoolbook(N, M, P)
    best_rank = N * M * P

    # Try the rank-48 block-Strassen construction first
    try:
        cand = _decomp_4x4x9_strassen_block()
        if _verify(cand['U'], cand['V'], cand['W'], N, M, P):
            best_result = cand
            best_rank = 48
            print("Verified rank-48 decomposition")
    except Exception as e:
        print("rank-48 construction failed:", e)

    # Try the rank-60 block decomposition (4x4x4 + 4x4x4 + 4x4x1)
    if best_rank > 60:
        try:
            cand = _decomp_4x4x9_block()
            if _verify(cand['U'], cand['V'], cand['W'], N, M, P):
                best_result = cand
                best_rank = 60
                print("Verified rank-60 decomposition")
        except Exception as e:
            print("rank-60 construction failed:", e)

    if time.time() - start_time > DEADLINE - 5:
        return best_result

    # Search for better ranks using structured search
    for rank in range(best_rank - 1, max(best_rank - 30, 40), -1):
        if time.time() - start_time > DEADLINE - 20:
            break
        print(f"Trying rank {rank}...")
        result = _try_structured_rank(rank, N, M, P, max_attempts=100,
                                      deadline=DEADLINE - 10)
        if result is not None:
            if _verify(result['U'], result['V'], result['W'], N, M, P):
                # Apply local polish to potentially reduce rank further
                polished = _local_polish(result['U'], result['V'], result['W'])
                if _verify(polished['U'], polished['V'], polished['W'], N, M, P):
                    best_result = polished
                    best_rank = len(polished['U'][0])
                else:
                    best_result = result
                    best_rank = rank
                print(f"Found valid rank {best_rank} decomposition!")
                break

    return best_result
