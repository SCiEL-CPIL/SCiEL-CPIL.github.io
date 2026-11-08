import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 120
    
    # Try rank 21 first
    for target_rank in [21, 22]:
        rank = target_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        def u_idx(i, k): return i * m + k
        def v_idx(k, j): return k * p + j
        def w_idx(j, i): return j * n + i

        if rank == 22:
            # Original rank-22 decomposition
            add_term([(u_idx(0,0), 1), (u_idx(1,1), 1)], [(v_idx(0,0), 1), (v_idx(1,1), 1)], [(w_idx(0,0), 1), (w_idx(1,1), 1)], 0)
            add_term([(u_idx(1,0), 1), (u_idx(1,1), 1)], [(v_idx(0,0), 1)], [(w_idx(0,1), 1), (w_idx(1,1), -1)], 1)
            add_term([(u_idx(0,0), 1)], [(v_idx(0,1), 1), (v_idx(1,1), -1)], [(w_idx(1,0), 1), (w_idx(1,1), 1)], 2)
            add_term([(u_idx(1,1), 1)], [(v_idx(1,0), 1), (v_idx(0,0), -1)], [(w_idx(0,0), 1), (w_idx(0,1), 1)], 3)
            add_term([(u_idx(0,0), 1), (u_idx(0,1), 1)], [(v_idx(1,1), 1)], [(w_idx(0,0), -1), (w_idx(1,0), 1)], 4)
            add_term([(u_idx(1,0), 1), (u_idx(0,0), -1)], [(v_idx(0,0), 1), (v_idx(0,1), 1)], [(w_idx(1,1), 1)], 5)
            add_term([(u_idx(0,1), 1), (u_idx(1,1), -1)], [(v_idx(1,0), 1), (v_idx(1,1), 1)], [(w_idx(0,0), 1)], 6)

            add_term([(u_idx(0,0), 1), (u_idx(1,1), 1)], [(v_idx(0,2), 1), (v_idx(1,3), 1)], [(w_idx(2,0), 1), (w_idx(3,1), 1)], 7)
            add_term([(u_idx(1,0), 1), (u_idx(1,1), 1)], [(v_idx(0,2), 1)], [(w_idx(2,1), 1), (w_idx(3,1), -1)], 8)
            add_term([(u_idx(0,0), 1)], [(v_idx(0,3), 1), (v_idx(1,3), -1)], [(w_idx(3,0), 1), (w_idx(3,1), 1)], 9)
            add_term([(u_idx(1,1), 1)], [(v_idx(1,2), 1), (v_idx(0,2), -1)], [(w_idx(2,0), 1), (w_idx(2,1), 1)], 10)
            add_term([(u_idx(0,0), 1), (u_idx(0,1), 1)], [(v_idx(1,3), 1)], [(w_idx(2,0), -1), (w_idx(3,0), 1)], 11)
            add_term([(u_idx(1,0), 1), (u_idx(0,0), -1)], [(v_idx(0,2), 1), (v_idx(0,3), 1)], [(w_idx(3,1), 1)], 12)
            add_term([(u_idx(0,1), 1), (u_idx(1,1), -1)], [(v_idx(1,2), 1), (v_idx(1,3), 1)], [(w_idx(2,0), 1)], 13)

            add_term([(u_idx(0,2), 1)], [(v_idx(2,0), 1)], [(w_idx(0,0), 1)], 14)
            add_term([(u_idx(0,2), 1)], [(v_idx(2,1), 1)], [(w_idx(1,0), 1)], 15)
            add_term([(u_idx(1,2), 1)], [(v_idx(2,0), 1)], [(w_idx(0,1), 1)], 16)
            add_term([(u_idx(1,2), 1)], [(v_idx(2,1), 1)], [(w_idx(1,1), 1)], 17)

            add_term([(u_idx(0,2), 1)], [(v_idx(2,2), 1)], [(w_idx(2,0), 1)], 18)
            add_term([(u_idx(0,2), 1)], [(v_idx(2,3), 1)], [(w_idx(3,0), 1)], 19)
            add_term([(u_idx(1,2), 1)], [(v_idx(2,2), 1)], [(w_idx(2,1), 1)], 20)
            add_term([(u_idx(1,2), 1)], [(v_idx(2,3), 1)], [(w_idx(3,1), 1)], 21)
        else:
            # Try rank 21: merge some outer product terms with Strassen terms
            # Modify Strassen terms to include k=2 components
            # Term 0: (A00+A11)*(B00+B11) -> C00+C11. Add A02,B20,C00? No, that changes the product.
            # Instead, try to use the structure differently.
            
            # Alternative: Use a different decomposition strategy
            # The 2x3 x 3x4 multiplication can be seen as:
            # C = A_left * B_left + A_right * B_right
            # where A_left is 2x2, B_left is 2x4, A_right is 2x1, B_right is 1x4
            
            # For rank 21, try to share between the two Strassen blocks
            # by using a modified Strassen for the combined problem
            
            # Let me try a direct construction with rank 21
            # This is a heuristic attempt - may not work
            
            # Part 1: Modified Strassen for A[:,:2]*B[:2,:2] with shared structure
            add_term([(u_idx(0,0), 1), (u_idx(1,1), 1), (u_idx(0,2), 1)], [(v_idx(0,0), 1), (v_idx(1,1), 1), (v_idx(2,2), 1)], [(w_idx(0,0), 1), (w_idx(1,1), 1)], 0)
            add_term([(u_idx(1,0), 1), (u_idx(1,1), 1)], [(v_idx(0,0), 1)], [(w_idx(0,1), 1), (w_idx(1,1), -1)], 1)
            add_term([(u_idx(0,0), 1)], [(v_idx(0,1), 1), (v_idx(1,1), -1)], [(w_idx(1,0), 1), (w_idx(1,1), 1)], 2)
            add_term([(u_idx(1,1), 1)], [(v_idx(1,0), 1), (v_idx(0,0), -1)], [(w_idx(0,0), 1), (w_idx(0,1), 1)], 3)
            add_term([(u_idx(0,0), 1), (u_idx(0,1), 1)], [(v_idx(1,1), 1)], [(w_idx(0,0), -1), (w_idx(1,0), 1)], 4)
            add_term([(u_idx(1,0), 1), (u_idx(0,0), -1)], [(v_idx(0,0), 1), (v_idx(0,1), 1)], [(w_idx(1,1), 1)], 5)
            add_term([(u_idx(0,1), 1), (u_idx(1,1), -1)], [(v_idx(1,0), 1), (v_idx(1,1), 1)], [(w_idx(0,0), 1)], 6)

            # Part 2: Strassen for A[:,:2]*B[:2,2:4]
            add_term([(u_idx(0,0), 1), (u_idx(1,1), 1)], [(v_idx(0,2), 1), (v_idx(1,3), 1)], [(w_idx(2,0), 1), (w_idx(3,1), 1)], 7)
            add_term([(u_idx(1,0), 1), (u_idx(1,1), 1)], [(v_idx(0,2), 1)], [(w_idx(2,1), 1), (w_idx(3,1), -1)], 8)
            add_term([(u_idx(0,0), 1)], [(v_idx(0,3), 1), (v_idx(1,3), -1)], [(w_idx(3,0), 1), (w_idx(3,1), 1)], 9)
            add_term([(u_idx(1,1), 1)], [(v_idx(1,2), 1), (v_idx(0,2), -1)], [(w_idx(2,0), 1), (w_idx(2,1), 1)], 10)
            add_term([(u_idx(0,0), 1), (u_idx(0,1), 1)], [(v_idx(1,3), 1)], [(w_idx(2,0), -1), (w_idx(3,0), 1)], 11)
            add_term([(u_idx(1,0), 1), (u_idx(0,0), -1)], [(v_idx(0,2), 1), (v_idx(0,3), 1)], [(w_idx(3,1), 1)], 12)
            add_term([(u_idx(0,1), 1), (u_idx(1,1), -1)], [(v_idx(1,2), 1), (v_idx(1,3), 1)], [(w_idx(2,0), 1)], 13)

            # Part 3: A[:,2]*B[2,:2] - try to merge with existing terms
            # C00 += A02*B20, C10 += A02*B21, C01 += A12*B20, C11 += A12*B21
            # Try to add these to existing terms that already touch C00, C10, etc.
            # Term 0 already touches C00 and C11. Can I add A02*B20 to it? No, because it's a single product.
            # I need separate terms or to modify the U/V vectors.
            
            # Alternative: modify U[0,2] and V[2,0] in some terms
            # This gets complex. Let me just use 3 terms for Part 3 (merge one pair)
            # Actually, I can't easily merge outer products.
            
            # For now, use 3 terms for Part 3 (one less than 4)
            # This won't work in general. Let me just return rank 22 if 21 fails.
            
            # Use 4 terms for Part 3
            add_term([(u_idx(0,2), 1)], [(v_idx(2,0), 1)], [(w_idx(0,0), 1)], 14)
            add_term([(u_idx(0,2), 1)], [(v_idx(2,1), 1)], [(w_idx(1,0), 1)], 15)
            add_term([(u_idx(1,2), 1)], [(v_idx(2,0), 1)], [(w_idx(0,1), 1)], 16)
            add_term([(u_idx(1,2), 1)], [(v_idx(2,1), 1)], [(w_idx(1,1), 1)], 17)

            # Part 4: A[:,2]*B[2,2:4] - use 3 terms (save 1)
            add_term([(u_idx(0,2), 1)], [(v_idx(2,2), 1)], [(w_idx(2,0), 1)], 18)
            add_term([(u_idx(0,2), 1)], [(v_idx(2,3), 1)], [(w_idx(3,0), 1)], 19)
            add_term([(u_idx(1,2), 1)], [(v_idx(2,2), 1), (v_idx(2,3), 1)], [(w_idx(2,1), 1), (w_idx(3,1), 1)], 20)
            # This last term doesn't work because (A12)*(B22+B23) = A12*B22 + A12*B23, which is correct!
            # But wait, I need C21 += A12*B22 and C31 += A12*B23. The W term has both, so this works.
            # Actually no: W[2,1]=1 and W[3,1]=1, so we get A12*(B22+B23)*1 for C21 and A12*(B22+B23)*1 for C31.
            # That's wrong! We need C21 += A12*B22 (not A12*B23) and C31 += A12*B23 (not A12*B22).
            
            # So I can't merge them that way. Let me use 4 terms.
            # Total: 7+7+4+4 = 22. Can't get to 21 this way.
            
            # Let me just return rank 22
            pass
        
        # Verify
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m)).astype(float)
        B_test = np.random.randint(-5, 5, size=(m, p)).astype(float)
        C_expected = A_test @ B_test
        
        C_reconstructed = np.zeros((n, p), dtype=float)
        for i in range(n):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    u_prod = sum(U_np[i * m + k, r] * A_test[i, k] for k in range(m))
                    v_prod = sum(V_np[k * p + j, r] * B_test[k, j] for k in range(m))
                    val += W_np[j * n + i, r] * u_prod * v_prod
                C_reconstructed[i, j] = val
                
        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            if rank == 21:
                continue
            else:
                raise ValueError("Decomposition verification failed")
        
        return {"U": U, "V": V, "W": W}
    
    raise ValueError("No valid decomposition found")
