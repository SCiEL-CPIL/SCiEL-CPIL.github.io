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

    def snap_to_half_integer(values, tol=1e-6):
        """Snap values to nearest half-integer (multiple of 0.5)."""
        snapped = []
        for v in values:
            if abs(v) < tol:
                snapped.append(Fraction(0))
                continue
            # Try to snap to nearest multiple of 0.5
            nearest = round(2 * v) / 2
            if abs(v - nearest) < tol:
                snapped.append(Fraction(nearest))
            else:
                # Fall back to small rational
                best_frac = Fraction(0)
                best_dist = float('inf')
                for denom in range(1, 9):
                    num = round(v * denom)
                    frac = Fraction(num, denom)
                    dist = abs(v - float(frac))
                    if dist < best_dist:
                        best_dist = dist
                        best_frac = frac
                snapped.append(best_frac)
        return snapped

    def search_rank(R, max_iterations=300, max_restarts=15):
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
        
        # Precompute index arrays for vectorization
        ijs = []
        jks = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ijs.append(i * m + j)
                    jks.append(j * p + k)
        ijs_arr = np.array(ijs)
        jks_arr = np.array(jks)
        
        # Target matrix T (45, 15) - precompute
        T_target = np.zeros((45, num_w))
        for c in range(45):
            i_c = c // 15
            j_c = (c % 15) // 5
            k_c = c % 5
            target_d = k_c * n + i_c
            T_target[c, target_d] = 1.0
        
        def compute_error_grad(U, V, W):
            """
            U: (9, R), V: (15, R), W: (15, R)
            Returns (error, grad_U, grad_V, grad_W)
            """
            # S[c, r] = U[ijs[c], r] * V[jks[c], r], shape (45, R)
            S = U[ijs_arr, :] * V[jks_arr, :]
            
            # P[c, d] = sum_r S[c, r] * W[d, r] = (S @ W.T)[c, d], shape (45, 15)
            P = S @ W.T
            
            # Error: ||P - T_target||_F^2
            E = P - T_target
            err = np.sum(E ** 2)
            
            # Gradients
            # dErr/dW[d, r] = sum_c 2 * E[c, d] * S[c, r]
            # = 2 * (E.T @ S)[d, r]
            grad_W = 2.0 * (E.T @ S)
            
            # dErr/dU[i, r] = sum_{c: ijs[c]==i} sum_d 2*E[c,d] * V[jks[c], r] * W[d, r]
            # dErr/dV[j, r] = sum_{c: jks[c]==j} sum_d 2*E[c,d] * U[ijs[c], r] * W[d, r]
            
            # EW[c, r] = sum_d E[c, d] * W[d, r] = (E @ W)[c, r]
            EW = E @ W
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            
            # For each c, add to U[ijs[c], :] and V[jks[c], :]
            # dU[ijs[c], r] += 2 * EW[c, r] * V[jks[c], r]
            # dV[jks[c], r] += 2 * EW[c, r] * U[ijs[c], r]
            
            dU_contrib = 2.0 * EW * V[jks_arr, :]
            dV_contrib = 2.0 * EW * U[ijs_arr, :]
            
            np.add.at(grad_U, ijs_arr, dU_contrib)
            np.add.at(grad_V, jks_arr, dV_contrib)
            
            return err, grad_U, grad_V, grad_W
        
        def try_snap_and_verify(U, V, W):
            """Snap to half-integers and verify."""
            U_flat = U.flatten()
            V_flat = V.flatten()
            W_flat = W.flatten()
            
            U_snapped = snap_to_half_integer(U_flat.tolist())
            V_snapped = snap_to_half_integer(V_flat.tolist())
            W_snapped = snap_to_half_integer(W_flat.tolist())
            
            U_mat = [U_snapped[i*num_u//num_u : (i+1)*num_u//num_u] if False else U_snapped[i*R:(i+1)*R] for i in range(num_u)]
            V_mat = [V_snapped[i*R:(i+1)*R] for i in range(num_v)]
            W_mat = [W_snapped[i*R:(i+1)*R] for i in range(num_w)]
            
            if verify_decomposition(U_mat, V_mat, W_mat):
                return {"U": [float(x) for x in row] for row in U_mat}, \
                       {"V": [float(x) for x in row] for row in V_mat}, \
                       {"W": [float(x) for x in row] for row in W_mat}
            return None
        
        # Search Loop
        best_result = None
        best_err_global = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random Initialization with different strategies
            if restart % 3 == 0:
                U = np.random.uniform(-1, 1, (num_u, R))
                V = np.random.uniform(-1, 1, (num_v, R))
                W = np.random.uniform(-1, 1, (num_w, R))
            elif restart % 3 == 1:
                # Initialize with some structure
                U = np.zeros((num_u, R))
                V = np.zeros((num_v, R))
                W = np.zeros((num_w, R))
                for r in range(R):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                # Add random perturbation
                U += np.random.normal(0, 0.1, U.shape)
                V += np.random.normal(0, 0.1, V.shape)
                W += np.random.normal(0, 0.1, W.shape)
            else:
                # Sparse random initialization
                U = np.zeros((num_u, R))
                V = np.zeros((num_v, R))
                W = np.zeros((num_w, R))
                for r in range(R):
                    u_idx = random.randint(0, num_u-1)
                    v_idx = random.randint(0, num_v-1)
                    w_idx = random.randint(0, num_w-1)
                    U[u_idx, r] = random.choice([-1, 1])
                    V[v_idx, r] = random.choice([-1, 1])
                    W[w_idx, r] = random.choice([-1, 1])
            
            lr = 0.05
            best_err = float('inf')
            
            for it in range(max_iterations):
                if time.time() > deadline - 10:
                    break
                
                err, gU, gV, gW = compute_error_grad(U, V, W)
                
                if err < best_err:
                    best_err = err
                    if err < 1e-8:
                        # Try to snap and verify
                        result = try_snap_and_verify(U, V, W)
                        if result is not None:
                            return result
                
                # Adaptive learning rate based on gradient norm
                grad_norm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2))
                if grad_norm > 1e-10:
                    step = lr * err / (grad_norm + 1e-10)
                    step = min(step, 0.1)
                    U -= step * gU / grad_norm * 0.1
                    V -= step * gV / grad_norm * 0.1
                    W -= step * gW / grad_norm * 0.1
                else:
                    U -= lr * gU
                    V -= lr * gV
                    W -= lr * gW
                
                # Decay learning rate
                lr *= 0.995
                
                # Occasional noise to escape local minima
                if it % 30 == 0 and it > 0:
                    noise_scale = 0.05 * (1.0 + 0.5 * np.sin(it * 0.1))
                    U += np.random.normal(0, noise_scale, U.shape)
                    V += np.random.normal(0, noise_scale, V.shape)
                    W += np.random.normal(0, noise_scale, W.shape)
            
            # After each restart, try to snap the best found
            if best_err < 1e-6:
                result = try_snap_and_verify(U, V, W)
                if result is not None:
                    return result
        
        return None

    # Try ranks from low to high - start lower
    best_result = None
    for R in range(18, 45):
        if time.time() > deadline - 15:
            break
        result = search_rank(R, max_iterations=200, max_restarts=10)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-END
