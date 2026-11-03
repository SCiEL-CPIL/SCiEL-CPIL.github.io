import numpy as np
import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Generate all possible 2x2x2 blocks
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                I = (i0, i0 + 1)
                J = (j0, j0 + 1)
                K = (k0, k0 + 1)
                terms = set()
                for i in I:
                    for j in J:
                        for k in K:
                            terms.add((i, j, k))
                blocks.append((i0, j0, k0, I, J, K, terms))
    
    # Strassen coefficients for 2x2x2 block (local indices)
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Precompute Strassen vectors
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # M1
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
    
    # M2
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[1, 1] = 1
    W_strassen[3, 1] = -1
    
    # M3
    U_strassen[0, 2] = 1
    V_strassen[1, 2] = 1; V_strassen[3, 2] = -1
    W_strassen[2, 2] = 1
    W_strassen[3, 2] = 1
    
    # M4
    U_strassen[3, 3] = 1
    V_strassen[2, 3] = 1; V_strassen[0, 3] = -1
    W_strassen[0, 3] = 1
    W_strassen[1, 3] = 1
    
    # M5
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[0, 4] = -1
    W_strassen[2, 4] = 1
    
    # M6
    U_strassen[2, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1
    W_strassen[3, 5] = 1
    
    # M7
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1
    W_strassen[0, 6] = 1
    
    def build_decomposition(selected_blocks):
        """Build U, V, W from selected Strassen blocks + schoolbook for rest."""
        num_blocks = len(selected_blocks)
        remaining_terms = n * m * p - num_blocks * 8
        rank = num_blocks * 7 + remaining_terms
        
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        covered = set()
        
        # Add Strassen blocks
        for (i0, j0, k0, I, J, K, terms) in selected_blocks:
            for t in range(7):
                for li in range(2):
                    for lj in range(2):
                        val = U_strassen[li*2+lj, t]
                        if val != 0:
                            gi = i0 if li == 0 else i0 + 1
                            gj = j0 if lj == 0 else j0 + 1
                            U[gi*m + gj, col] += val
                
                for lj in range(2):
                    for lk in range(2):
                        val = V_strassen[lj*2+lk, t]
                        if val != 0:
                            gj = j0 if lj == 0 else j0 + 1
                            gk = k0 if lk == 0 else k0 + 1
                            V[gj*p + gk, col] += val
                
                for lk in range(2):
                    for li in range(2):
                        val = W_strassen[lk*2+li, t]
                        if val != 0:
                            gk = k0 if lk == 0 else k0 + 1
                            gi = i0 if li == 0 else i0 + 1
                            W[gk*n + gi, col] += val
                
                col += 1
            covered |= terms
        
        # Add remaining schoolbook terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
        
        if col < U.shape[1]:
            U = U[:, :col]
            V = V[:, :col]
            W = W[:, :col]
            
        return U, V, W
    
    def verify(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            T += np.einsum('ij,jk,ki->ijk', U_r[:, :, c], V_r[:, :, c], W_r[:, :, c])
            
        return np.allclose(T, 1.0, atol=1e-9)
    
    # Greedy with scoring: prefer blocks that cover more uncovered terms
    best_blocks = []
    best_count = 0
    
    # Randomized restarts with different orderings
    num_restarts = 300
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Shuffle with slight bias toward certain positions
        shuffled_blocks = blocks[:]
        random.shuffle(shuffled_blocks)
        
        covered = set()
        current_blocks = []
        
        for block in shuffled_blocks:
            i0, j0, k0, I, J, K, terms = block
            new_coverage = terms - covered
            if len(new_coverage) > 0:
                covered |= terms
                current_blocks.append(block)
                
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]
            
            # Try local search: swap one block for another
            improved = True
            while improved:
                improved = False
                for idx in range(len(best_blocks)):
                    old_block = best_blocks[idx]
                    # Remove old block
                    temp_covered = set()
                    for b in best_blocks:
                        if b != old_block:
                            temp_covered |= b[6]
                    
                    # Try adding each unselected block
                    for new_block in blocks:
                        if new_block in best_blocks:
                            continue
                        if new_block[6] & temp_covered:
                            continue
                        new_covered = temp_covered | new_block[6]
                        # Check if this improves (same count but maybe better for future)
                        # For now, just accept if same count
                        if len(new_covered) >= len(covered):
                            best_blocks[idx] = new_block
                            covered = new_covered
                            improved = True
                            break
                    if improved:
                        break
    
    # Build initial decomposition
    U, V, W = build_decomposition(best_blocks)
    
    if not verify(U, V, W):
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    current_rank = U.shape[1]
    
    # Attempt rank reduction via linear algebra
    # Check if any rank-1 tensor is in the span of others
    # This is expensive, so only try if we have time
    
    if time.time() < deadline - 30 and current_rank > 90:
        try:
            # Try to find if removing one column still gives valid decomposition
            # by checking if that column is linearly dependent in a specific way
            # Simplified: just try random subsets and verify
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                    
                # Try removing a random column
                col_to_remove = random.randint(0, U.shape[1] - 1)
                mask = [i for i in range(U.shape[1]) if i != col_to_remove]
                
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]
                
                if verify(U_test, V_test, W_test):
                    U, V, W = U_test, V_test, W_test
                    current_rank = U.shape[1]
        except Exception:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
