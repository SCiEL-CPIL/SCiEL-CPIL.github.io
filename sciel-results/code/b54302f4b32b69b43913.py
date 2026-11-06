import numpy as np
import time

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy
    sm = sx - 1j * sy
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]
    
    def bond_hamiltonian(i, j):
        sp_i = op_on_site(sp, i)
        sm_j = op_on_site(sm, j)
        sm_i = op_on_site(sm, i)
        sp_j = op_on_site(sp, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return gamma / 2 * (sp_i @ sm_j + sm_i @ sp_j) + sz_i @ sz_j
    
    for i, j in odd_bonds:
        A += bond_hamiltonian(i, j)
    for i, j in even_bonds:
        B += bond_hamiltonian(i, j)
    
    half_field = 0.5 * h
    for s in range(N):
        sz_s = op_on_site(sz, s)
        A -= half_field * sz_s
        B -= half_field * sz_s
    
    return A, B

def coefficients(stages):
    # 4th-order Yoshida coefficients for 5-step A, 3-step B scheme
    c = 2.0 ** (1.0/3.0)
    c1 = 1.0 / (2.0 - c)
    c2 = -c / (2.0 - c)
    
    a_1 = c1
    a_2 = c1
    a_3 = 1.0 - 2.0 * c1
    a_4 = c1
    a_5 = c1
    
    b_1 = c2
    b_2 = 1.0 - 2.0 * c2
    b_3 = c2
    
    # Map to 12 stages
    a = [0.0] * stages
    b = [0.0] * stages
    
    # Center the 5 A's and 3 B's
    # A: indices 3, 4, 5, 6, 7
    # B: indices 4, 5, 6
    a[3] = a_1
    a[4] = a_2
    a[5] = a_3
    a[6] = a_4
    a[7] = a_5
    
    b[4] = b_1
    b[5] = b_2
    b[6] = b_3
    
    # Normalize to ensure sum is 1
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    
    # Clip to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Re-normalize
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
        
    return a, b
