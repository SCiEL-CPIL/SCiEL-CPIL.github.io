import time
import numpy as np
from fractions import Fraction

N, M, P = 3, 4, 4
NM, MP, PN = N * M, M * P, P * N


def _verify(U, V, W, tol=1e-10):
    """Verify using einsum."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err


def _schoolbook():
    rank = NM * MP * PN
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {'U': U, 'V': V, 'W': W}


def _strassen_44():
    rank = 44
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    # 7 Strassen steps for 2x2 block
    s_steps = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 1): 1, (0, 0): -1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1})
    ]
    
    # 4 blocks for the 2x2 output blocks
    blocks = [(0, 0, 0, 0, 0, 0), (0, 2, 2, 0, 0, 0), (0, 0, 0, 2, 0, 2), (0, 2, 2, 2, 0, 2)]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for (u_c, v_c, w_c) in s_steps:
            for (a, b), val in u_c.items():
                U[(a + xi) * M + (b + xj)][col] += val
            for (b, c), val in v_c.items():
                V[(b + yj) * P + (c + yk)][col] += val
            for (a, c), val in w_c.items():
                W[(c + zk) * N + (a + zi)][col] += val
            col += 1
    
    # Bottom row (row 2) done naively: 4x4 = 16 terms
    for j in range(M):
        for k in range(P):
            U[2 * M + j][col] = 1
            V[j * P + k][col] = 1
            W[k * N + 2][col] = 1
            col += 1
    
    assert col == rank
    return {'U': U, 'V': V, 'W': W}


def _verify_fast(U, V, W):
    """Faster verification that returns just the error."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    return np.max(np.abs(T - 1.0))


def _try_reduce_by_merge(U, V, W, deadline):
    """
    Try to reduce rank by finding pairs of columns that can be merged.
    Two columns c1, c2 can be merged if their product tensors are linearly dependent.
    This is a heuristic: we check if the outer product structure allows merging.
    """
    rank = len(U[0])
    best_U, best_V, best_W = U[:], V[:], W[:]
    best_rank = rank
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    deadline = deadline - 5.0  # safety margin
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Try to find a pair of columns that can be merged
        # Check if column c2 is a scalar multiple of column c1 in the tensor sense
        # This means there exists alpha such that for all i,j,k:
        # U_i,c2 * V_j,c2 * W_k,c2 = alpha * (U_i,c1 * V_j,c1 * W_k,c1)
        # This is equivalent to: U[:,c2] = a*U[:,c1], V[:,c2] = b*V[:,c1], W[:,c2] = c*W[:,c1]
        # where a*b*c = 1
        
        found_merge = False
        for c1 in range(best_rank - 1):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, best_rank):
                # Check if U[:,c2] is proportional to U[:,c1]
                # Find non-zero entries
                u1 = [U_frac[i][c1] for i in range(NM)]
                u2 = [U_frac[i][c2] for i in range(NM)]
                v1 = [V_frac[i][c1] for i in range(MP)]
                v2 = [V_frac[i][c2] for i in range(MP)]
                w1 = [W_frac[i][c1] for i in range(PN)]
                w2 = [W_frac[i][c2] for i in range(PN)]
                
                # Check proportionality
                a = None
                for i in range(NM):
                    if u1[i] != 0:
                        if u2[i] != 0 and a is None:
                            a = u2[i] / u1[i]
                        elif u2[i] == 0:
                            a = None
                            break
                        elif u2[i] / u1[i] != a:
                            a = None
                            break
                    elif u2[i] != 0:
                        a = None
                        break
                
                if a is None or a == 0:
                    continue
                
                b = None
                for i in range(MP):
                    if v1[i] != 0:
                        if v2[i] != 0 and b is None:
                            b = v2[i] / v1[i]
                        elif v2[i] == 0:
                            b = None
                            break
                        elif v2[i] / v1[i] != b:
                            b = None
                            break
                    elif v2[i] != 0:
                        b = None
                        break
                
                if b is None or b == 0:
                    continue
                
                c = None
                for i in range(PN):
                    if w1[i] != 0:
                        if w2[i] != 0 and c is None:
                            c = w2[i] / w1[i]
                        elif w2[i] == 0:
                            c = None
                            break
                        elif w2[i] / w1[i] != c:
                            c = None
                            break
                    elif w2[i] != 0:
                        c = None
                        break
                
                if c is None or c == 0:
                    continue
                
                # Check if a*b*c = 1
                if a * b * c != 1:
                    continue
                
                # Found a merge! Remove column c2
                # New column c1' = U[:,c1] + U[:,c2], etc.
                for i in range(NM):
                    U_frac[i][c1] = U_frac[i][c1] + U_frac[i][c2]
                for i in range(MP):
                    V_frac[i][c1] = V_frac[i][c1] + V_frac[i][c2]
                for i in range(PN):
                    W_frac[i][c1] = W_frac[i][c1] + W_frac[i][c2]
                
                # Remove column c2
                del U_frac[0][c2]  # placeholder, need to delete from all rows
                # Actually, let's rebuild
                U_frac = [row[:c2] + row[c2+1:] for row in U_frac]
                V_frac = [row[:c2] + row[c2+1:] for row in V_frac]
                W_frac = [row[:c2] + row[c2+1:] for row in W_frac]
                
                best_rank = best_rank - 1
                best_U = [[float(x) for x in row] for row in U_frac]
                best_V = [[float(x) for x in row] for row in V_frac]
                best_W = [[float(x) for x in row] for row in W_frac]
                
                found_merge = True
                break
            if found_merge or time.time() > deadline:
                break
        
        if not found_merge:
            break
    
    return {'U': best_U, 'V': best_V, 'W': best_W}, best_rank


def _gaussian_reduce(U, V, W, deadline):
    """
    Try to reduce rank using Gaussian elimination on the tensor.
    This is a heuristic approach: we try to find linear dependencies among the columns.
    """
    rank = len(U[0])
    best_U, best_V, best_W = U[:], V[:], W[:]
    best_rank = rank
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    deadline = deadline - 5.0  # safety margin
    
    # Try to find columns that are linear combinations of others
    # This is complex, so we'll use a simpler heuristic:
    # Check if any two columns have the same support pattern
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        found_reduction = False
        for c1 in range(best_rank - 1):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, best_rank):
                # Check if column c2 can be expressed as a linear combination of column c1
                # and possibly others. For simplicity, we only check if they are identical
                # or negatives of each other.
                
                u1 = [U_frac[i][c1] for i in range(NM)]
                u2 = [U_frac[i][c2] for i in range(NM)]
                v1 = [V_frac[i][c1] for i in range(MP)]
                v2 = [V_frac[i][c2] for i in range(MP)]
                w1 = [W_frac[i][c1] for i in range(PN)]
                w2 = [W_frac[i][c2] for i in range(PN)]
                
                # Check if u2 = -u1, v2 = -v1, w2 = w1 (or similar)
                if all(u1[i] + u2[i] == 0 for i in range(NM)) and \
                   all(v1[i] + v2[i] == 0 for i in range(MP)) and \
                   all(w1[i] - w2[i] == 0 for i in range(PN)):
                    # u2 = -u1, v2 = -v1, w2 = w1
                    # Product: u2*v2*w2 = (-u1)*(-v1)*w1 = u1*v1*w1
                    # So we can remove column c2 by adding column c1 to it
                    # Actually, the products are equal, so we can just remove c2
                    # and keep c1 as is
                    del U_frac[0][c2]
                    U_frac = [row[:c2] + row[c2+1:] for row in U_frac]
                    V_frac = [row[:c2] + row[c2+1:] for row in V_frac]
                    W_frac = [row[:c2] + row[c2+1:] for row in W_frac]
                    
                    best_rank = best_rank - 1
                    best_U = [[float(x) for x in row] for row in U_frac]
                    best_V = [[float(x) for x in row] for row in V_frac]
                    best_W = [[float(x) for x in row] for row in W_frac]
                    
                    found_reduction = True
                    break
            if found_reduction or time.time() > deadline:
                break
        
        if not found_reduction:
            break
    
    return {'U': best_U, 'V': best_V, 'W': best_W}, best_rank


def construct():
    start = time.time()
    
    # Start with the known Strassen-based decomposition with rank 44
    result = _strassen_44()
    
    # Verify correctness
    ok, err = _verify(result['U'], result['V'], result['W'])
    if not ok:
        # Fallback to schoolbook (should never happen)
        result = _schoolbook()
        return result
    
    best_result = result
    best_rank = len(result['U'][0])
    
    # Deadline: 140 seconds (10s safety margin)
    deadline = start + 140.0
    
    # Try to reduce rank using merge heuristic
    if time.time() < deadline - 30:
        merged_result, merged_rank = _try_reduce_by_merge(result['U'], result['V'], result['W'], deadline)
        if merged_rank < best_rank:
            # Verify the merged result
            ok, err = _verify(merged_result['U'], merged_result['V'], merged_result['W'])
            if ok:
                best_result = merged_result
                best_rank = merged_rank
    
    # Try Gaussian reduction
    if time.time() < deadline - 10:
        reduced_result, reduced_rank = _gaussian_reduce(best_result['U'], best_result['V'], best_result['W'], deadline)
        if reduced_rank < best_rank:
            # Verify the reduced result
            ok, err = _verify(reduced_result['U'], reduced_result['V'], reduced_result['W'])
            if ok:
                best_result = reduced_result
                best_rank = reduced_rank
    
    # Final verification
    ok, err = _verify(best_result['U'], best_result['V'], best_result['W'])
    if not ok:
        # Fallback to the original Strassen result
        best_result = result
    
    return best_result
