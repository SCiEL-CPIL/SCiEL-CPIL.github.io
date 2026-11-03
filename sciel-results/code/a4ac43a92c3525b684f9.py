import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    num_u = n * m  # 9
    num_v = m * p  # 15
    num_w = p * n  # 15
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(num_u)]
        V = [[0] * rank for _ in range(num_v)]
        W = [[0] * rank for _ in range(num_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication.
        Returns True if valid, False otherwise."""
        rank = len(U[0])
        
        # Check all (i, j, k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The term U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should sum to 1
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if total != 1:
                        return False
        
        # Check that no spurious terms exist
        # For any (i, j, k) and (i', k') where (i', k') != (i, k),
        # the sum should be 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            if i2 == i and k2 == k:
                                continue
                            total = 0
                            for r in range(rank):
                                total += U[i * m + j][r] * V[j * p + k][r] * W[k2 * n + i2][r]
                            if total != 0:
                                return False
        return True

    def snap_to_rational(values, max_denom=8):
        """Snap values to nearest fraction with small denominator."""
        snapped = []
        for v in values:
            if abs(v) < 1e-9:
                snapped.append(Fraction(0))
                continue
            best_frac = Fraction(0)
            best_dist = float('inf')
            for denom in range(1, max_denom + 1):
                num = round(v * denom)
                frac = Fraction(num, denom)
                dist = abs(v - float(frac))
                if dist < best_dist:
                    best_dist = dist
                    best_frac = frac
            snapped.append(best_frac)
        return snapped

    def search_rank(R, max_iterations=200, max_restarts=10):
        """Try to find a rank-R decomposition using randomized local search."""
        if time.time() > deadline - 10:
            return None
        
        # Pre-compute indices for vectorization
        # We want to compute T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Target: T[i, j, k] = 1 if (i, j, k) corresponds to (i, j, k) in matmul?
        # Wait, the tensor is M_{ijk} = delta_{i, i} delta_{j, j} delta_{k, k}?
        # No, matrix multiplication C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor T has indices i (row of A), j (col of A/row of B), k (col of B).
        # T[i, j, k] = 1 if we are computing C_{i, k} using A_{i, j} B_{j, k}.
        # So T[i, j, k] = 1 for all i, j, k?
        # Let's check the definition in verify_decomposition.
        # "The term U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should sum to 1"
        # This implies that for EVERY (i, j, k), the sum is 1.
        # And for spurious terms: "For any (i, j, k) and (i', k') where (i', k') != (i, k)"
        # Wait, the spurious term check in the original code is:
        # sum_r U[i, j, r] V[j, k, r] W[k2, i2, r] == 0 for (i2, k2) != (i, k).
        # This is checking if the decomposition is "diagonal" in the output space?
        # No, matrix multiplication is linear.
        # C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # If we write C = sum_r U_r V_r W_r^T (outer products),
        # Then C_{i, k} = sum_r (sum_j U_{i, j}^{(r)} V_{j, k}^{(r)}) W_{i, k}^{(r)}?
        # No, the standard tensor decomposition is:
        # T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}.
        # For matrix multiplication, T_{i, j, k} = 1 for all i, j, k?
        # Let's check: C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # The tensor of multiplication is defined such that T(A, B) = C.
        # T is a trilinear form? No, it's a tensor in $V^* \otimes W^* \otimes Z$.
        # The entries are $T_{i, j, k} = 1$ if we consider the basis vectors $e_i, e_j, e_k$.
        # $M(e_i, e_j) = e_k$? No.
        # $M(e_i, e_j) = 0$ unless $i$ is row, $j$ is col?
        # Actually, the tensor $T$ has $T_{i, j, k} = 1$ for all $i, j, k$ is WRONG.
        # Let's look at the schoolbook code:
        # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1.
        # Sum_r U[i,j,r] V[j,k,r] W[k',i',r].
        # If we set k'=k and i'=i, we get sum_r 1*1*1 = 1.
        # If we set k'!=k or i'!=i, W[k',i',r] is 0 for the specific r where U and V are 1?
        # In schoolbook, each r corresponds to a unique (i,j,k).
        # U[i,j,r] is 1 only if r corresponds to (i,j,k).
        # V[j,k,r] is 1 only if r corresponds to (i,j,k).
        # W[k,i,r] is 1 only if r corresponds to (i,j,k).
        # So for a fixed (i,j,k), only one r is non-zero in all three.
        # If we check spurious term for (i2, k2) != (i, k):
        # We sum over r. For r corresponding to (i,j,k), U[i,j,r]=1, V[j,k,r]=1, but W[k2,i2,r]=0 (since k2!=k or i2!=i).
        # So the sum is 0.
        # This confirms the tensor structure:
        # We want $T_{i, j, k} = 1$ for all $i, j, k$?
        # Wait, if $T_{i, j, k} = 1$ for all $i, j, k$, then the "spurious" check is redundant?
        # No, the "spurious" check in the code is:
        # sum_r U[i, j, r] V[j, k, r] W[k2, i2, r] == 0.
        # This is NOT the same as T_{i, j, k} = 1.
        # This is checking that the "output" at (i2, k2) is 0 when we feed in inputs that would produce a non-zero at (i, k)?
        # No, the indices of U, V, W are fixed by the input positions.
        # The decomposition defines a map:
        # Given A (indexed by i, j) and B (indexed by j, k),
        # C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # The tensor decomposition $T = sum_r u_r \otimes v_r \otimes w_r$
        # means $C_{i, k} = sum_j sum_r A_{i, j} u_r(i, j) B_{j, k} v_r(j, k) w_r(i, k)$?
        # No, the standard form is:
        # $C_{i, k} = sum_r (sum_j U_{i, j}^{(r)} A_{i, j}) (sum_k V_{j, k}^{(r)} B_{j, k}) W_{i, k}^{(r)}$?
        # No, that's not right either.
        
        # Let's stick to the verification logic provided in the seed code, as it defines the task.
        # The seed code checks:
        # 1. sum_r U[i, j, r] V[j, k, r] W[k, i, r] == 1 for all i, j, k.
        # 2. sum_r U[i, j, r] V[j, k, r] W[k2, i2, r] == 0 for (i2, k2) != (i, k).
        
        # This implies that the tensor $T$ defined by $T_{i, j, k} = 1$ for all $i, j, k$ is NOT the target.
        # The target is a specific structure where the "output" indices (i, k) are decoupled from the "sum" index j in a specific way.
        # Actually, condition 1 says: For every (i, j, k), the product sums to 1.
        # Condition 2 says: For every (i, j, k) and (i2, k2) != (i, k), the product sums to 0.
        # This effectively means that the matrix $M_{(i, k), (i, j, k)}$ is the identity?
        # No, it means that the vector $y_{i, k} = sum_r (sum_j U_{i, j}^{(r)} V_{j, k}^{(r)}) W_{i, k}^{(r)}$?
        # No, the summation over r is outside.
        # Let $S_{i, j, k}^{(r)} = U_{i, j}^{(r)} V_{j, k}^{(r)}$.
        # Then condition 1: $sum_r S_{i, j, k}^{(r)} W_{k, i}^{(r)} = 1$.
        # Condition 2: $sum_r S_{i, j, k}^{(r)} W_{k2, i2}^{(r)} = 0$.
        
        # This is a very specific constraint.
        # Note that $U$ and $V$ do not depend on $i2, k2$.
        # This looks like we are trying to represent the identity map or something similar.
        
        # Let's just optimize for the error defined by these conditions.
        
        # Vectorized Error Calculation
        # We have U (9xR), V (15xR), W (15xR).
        # We need to check 3*3*5 = 45 conditions for "1" and 45 * (15-1) = 630 conditions for "0".
        # Total 675 constraints.
        
        def compute_error_grad(U, V, W):
            """
            U: (9, R)
            V: (15, R)
            W: (15, R)
            Returns error and gradients.
            """
            R = U.shape[1]
            # Flatten indices
            # U[i, j] where i in 0..2, j in 0..2. Index = i*3 + j.
            # V[j, k] where j in 0..2, k in 0..4. Index = j*5 + k.
            # W[k, i] where k in 0..4, i in 0..2. Index = k*3 + i.
            
            # Precompute all (i, j, k) tuples
            ijs = []
            jks = []
            kis = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ijs.append(i * m + j)
                        jks.append(j * p + k)
                        kis.append(k * n + i)
            
            ijs = np.array(ijs)
            jks = np.array(jks)
            kis = np.array(kis)
            
            # S_{i, j, k}^{(r)} = U[i, j, r] * V[j, k, r]
            # Shape: (45, R)
            S = U[ijs, :] * V[jks, :]
            
            # Target 1: sum_r S[:, r] * W[kis, r] == 1
            # W_kis = W[kis, :] shape (45, R)
            W_kis = W[kis, :]
            prod1 = S * W_kis
            sum1 = np.sum(prod1, axis=1) # Shape (45,)
            err1 = np.sum((sum1 - 1) ** 2)
            
            # Gradients for err1
            # d(err1)/d(W[kis, r]) = 2 * (sum1 - 1) * S[:, r]
            grad_W = np.zeros_like(W)
            # Accumulate gradients for W
            # For each constraint c (index in 0..44), the affected W index is kis[c]
            # dErr/dW[kis[c], r] += 2 * (sum1[c] - 1) * S[c, r]
            # We can use np.add.at
            delta1 = 2 * (sum1 - 1)[:, None] * S # Shape (45, R)
            np.add.at(grad_W, kis, delta1)
            
            # dErr/dU[ijs[c], r] += 2 * (sum1[c] - 1) * V[jks[c], r] * W[kis[c], r]
            # dErr/dV[jks[c], r] += 2 * (sum1[c] - 1) * U[ijs[c], r] * W[kis[c], r]
            
            term_W = W_kis
            term_V = V[jks, :]
            term_U = U[ijs, :]
            
            # Note: The factor 2 * (sum1 - 1) is common
            factor1 = 2 * (sum1 - 1)[:, None] # (45, 1)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            
            dU1 = factor1 * term_V * term_W
            dV1 = factor1 * term_U * term_W
            
            np.add.at(grad_U, ijs, dU1)
            np.add.at(grad_V, jks, dV1)
            
            # Target 2: Spurious terms
            # For each (i, j, k) and (i2, k2) != (i, k), sum_r S[i, j, k, r] * W[k2, i2, r] == 0
            # This is equivalent to: The matrix M where M_{c, d} = sum_r S[c, r] W[d, r] should be:
            # M[c, c] = 1 (from target 1, assuming c maps to (i,k))
            # M[c, d] = 0 for c != d.
            # Wait, the mapping is:
            # c corresponds to (i, j, k). The "output" index is (i, k).
            # d corresponds to (i2, k2).
            # The condition is that for a fixed input (i, j, k), the output at (i, k) is 1, and at all other (i2, k2) is 0.
            # This implies that the "output" is uniquely determined by (i, k).
            # But (i, j, k) determines (i, k).
            # So we have 45 inputs (i, j, k) and 15 outputs (i, k).
            # Multiple inputs map to the same output?
            # (i, j, k) -> (i, k).
            # For fixed i, k, there are m=3 values of j.
            # So 3 inputs map to the same output (i, k).
            # Condition:
            # For input (i, j, k), output (i, k) is 1.
            # For input (i, j, k), output (i2, k2) is 0 if (i2, k2) != (i, k).
            
            # Let's construct the matrix $P$ of shape (45, 15) where $P_{c, d} = sum_r S[c, r] W[d, r]$.
            # We want $P_{c, d} = 1$ if $d$ corresponds to $(i, k)$ of $c$, else 0.
            
            # Indices for W:
            # d in 0..14. d = k2 * 3 + i2.
            # For c (i, j, k), the target d is k * 3 + i.
            
            # Let's compute P = S @ W.T ?
            # S is (45, R). W is (15, R).
            # P_{c, d} = sum_r S[c, r] W[d, r] = (S @ W.T)[c, d].
            
            P = S @ W.T # Shape (45, 15)
            
            # Target matrix T (45, 15)
            T = np.zeros((45, 15))
            for c in range(45):
                i = (c // 5) // 3 # Wait, c is index in ijs list.
                # ijs[c] = i*3+j. jks[c] = j*5+k.
                # We need to recover i, j, k from c.
                # c goes 0..44.
                # i = c // 15? No.
                # Loop order: i, j, k.
                # c = i * 15 + j * 5 + k.
                i_c = c // 15
                j_c = (c % 15) // 5
                k_c = c % 5
                target_d = k_c * 3 + i_c
                T[c, target_d] = 1
            
            err2 = np.sum((P - T) ** 2)
            
            # Gradients for err2
            # d(err2)/d(W[d, r]) = sum_c 2 * (P[c, d] - T[c, d]) * S[c, r]
            # d(err2)/d(U[i, r]) ...
            # d(err2)/d(V[j, r]) ...
            
            # Let E = 2 * (P - T) # Shape (45, 15)
            E = 2 * (P - T)
            
            # dErr/dW[d, r] = sum_c E[c, d] S[c, r]
            # This is E.T @ S # Shape (15, R)
            dW2 = E.T @ S
            grad_W += dW2
            
            # dErr/dU[i, r]:
            # P[c, d] = sum_r' U[i_c, j_c, r'] V[j_c, k_c, r'] W[d, r']
            # dP[c, d]/dU[i, r] is non-zero only if c corresponds to (i, j_c, k_c) for some j_c, k_c?
            # No, U is indexed by (i, j).
            # For a fixed c, U[i_c, j_c, r] is the only U entry involved.
            # So dP[c, d]/dU[i, j, r] = V[j, k_c, r] W[d, r] if i_c==i and j_c==j, else 0.
            # dErr/dU[i, j, r] = sum_c sum_d E[c, d] * (V[j, k_c, r] W[d, r] * delta(i_c, i) delta(j_c, j))
            # = sum_{c: i_c==i, j_c==j} sum_d E[c, d] V[j, k_c, r] W[d, r]
            
            # We can compute this by looping over i, j or vectorizing.
            # Since dimensions are small, a loop over i, j (9 iterations) is fine.
            
            dU2 = np.zeros_like(U)
            dV2 = np.zeros_like(V)
            
            # Precompute V_part and W_part for each c?
            # V_part[c, r] = V[j_c, k_c, r]
            # W is already available.
            
            # For each c, we add to dU2[i_c, j_c, :] and dV2[j_c, k_c, :]
            # dU2[i_c, j_c, r] += sum_d E[c, d] V[j_c, k_c, r] W[d, r]
            # = (E[c, :] @ W) * V[j_c, k_c, r]  (element-wise product with V)
            
            # Let EW = E @ W # Shape (45, R)
            EW = E @ W
            
            # dU2[i_c, j_c, r] = EW[c, r] * V[j_c, k_c, r]
            # dV2[j_c, k_c, r] = EW[c, r] * U[i_c, j_c, r]
            
            # We need to scatter these back to U and V.
            # For each c, update U[i_c, j_c, :] and V[j_c, k_c, :]
            
            for c in range(45):
                i_c = c // 15
                j_c = (c % 15) // 5
                k_c = c % 5
                u_idx = i_c * 3 + j_c
                v_idx = j_c * 5 + k_c
                
                # EW[c, :] is shape (R,)
                # V[v_idx, :] is shape (R,)
                # U[u_idx, :] is shape (R,)
                
                dU2[u_idx, :] += EW[c, :] * V[v_idx, :]
                dV2[v_idx, :] += EW[c, :] * U[u_idx, :]
            
            grad_U += dU2
            grad_V += dV2
            
            total_error = err1 + err2
            return total_error, grad_U, grad_V, grad_W
        
        # Search Loop
        best_result = None
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random Initialization
            U = np.random.uniform(-1, 1, (num_u, R))
            V = np.random.uniform(-1, 1, (num_v, R))
            W = np.random.uniform(-1, 1, (num_w, R))
            
            lr = 0.01
            best_err = float('inf')
            
            for it in range(max_iterations):
                if time.time() > deadline - 10:
                    break
                
                err, gU, gV, gW = compute_error_grad(U, V, W)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-10:
                        # Snap to rational
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        
                        U_exact = [[Fraction(str(x)) for x in row] for row in U_list]
                        V_exact = [[Fraction(str(x)) for x in row] for row in V_list]
                        W_exact = [[Fraction(str(x)) for x in row] for row in W_list]
                        
                        if verify_decomposition(U_exact, V_exact, W_exact):
                            # Convert back to lists of floats/ints for return?
                            # The task asks for integer/half-integer entries.
                            # Fraction is fine, but let's ensure it's simple.
                            return {"U": U_list, "V": V_list, "W": W_list}
                
                # Update
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                # Decay learning rate
                lr *= 0.999
                
                # Occasional noise
                if it % 50 == 0:
                    U += np.random.normal(0, 0.01, U.shape)
                    V += np.random.normal(0, 0.01, V.shape)
                    W += np.random.normal(0, 0.01, W.shape)
                    
        return None

    # Try ranks from low to high
    best_result = None
    for R in range(25, 45):
        if time.time() > deadline - 15:
            break
        result = search_rank(R, max_iterations=100, max_restarts=5)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-END
