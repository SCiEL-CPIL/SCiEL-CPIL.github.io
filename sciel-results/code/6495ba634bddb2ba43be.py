"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product, combinations
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    n, m, p = 4, 5, 6
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24
    total_constraints = n * m * p  # 120

    # Generate candidate rank-1 terms with small integer entries
    # Each term is (U_pattern, V_pattern, W_pattern)
    # U: n x m, V: m x p, W: p x n
    # We use entries in {-1, 0, 1} for compactness

    def generate_candidates():
        """Generate diverse candidate rank-1 terms."""
        candidates = []
        
        # Pattern 1: Single 1 in each matrix (schoolbook-like)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U = np.zeros((n, m))
                    V = np.zeros((m, p))
                    W = np.zeros((p, n))
                    U[i, j] = 1
                    V[j, k] = 1
                    W[k, i] = 1
                    candidates.append((U, V, W))
        
        # Pattern 2: Row/column sums
        for i in range(n):
            U = np.zeros((n, m))
            U[i, :] = 1
            for k in range(p):
                V = np.zeros((m, p))
                V[:, k] = 1
                W = np.zeros((p, n))
                W[k, i] = 1
                candidates.append((U, V, W))
        
        for k in range(p):
            W = np.zeros((p, n))
            W[k, :] = 1
            for j in range(m):
                U = np.zeros((n, m))
                U[:, j] = 1
                V = np.zeros((m, p))
                V[j, k] = 1
                candidates.append((U, V, W))
        
        for j in range(m):
            V = np.zeros((m, p))
            V[j, :] = 1
            for i in range(n):
                U = np.zeros((n, m))
                U[i, j] = 1
                for k in range(p):
                    W = np.zeros((p, n))
                    W[k, i] = 1
                    candidates.append((U, V, W))
        
        # Pattern 3: Strassen-like combinations (2x2 blocks)
        # For pairs of rows/cols, use +-1 patterns
        for i1 in range(n):
            for i2 in range(i1+1, n):
                for j1 in range(m):
                    for j2 in range(j1+1, m):
                        for k1 in range(p):
                            for k2 in range(k1+1, p):
                                # Various sign patterns
                                for s1 in [1, -1]:
                                    for s2 in [1, -1]:
                                        for s3 in [1, -1]:
                                            U = np.zeros((n, m))
                                            V = np.zeros((m, p))
                                            W = np.zeros((p, n))
                                            U[i1, j1] = 1
                                            U[i1, j2] = s1
                                            U[i2, j1] = s2
                                            U[i2, j2] = 1
                                            V[j1, k1] = 1
                                            V[j1, k2] = s3
                                            V[j2, k1] = s1 * s2
                                            V[j2, k2] = 1
                                            W[k1, i1] = 1
                                            W[k1, i2] = s1
                                            W[k2, i1] = s2
                                            W[k2, i2] = 1
                                            candidates.append((U, V, W))
        
        # Pattern 4: Full row/col with alternating signs
        for i in range(n):
            for pattern in [[1,1,1,1,1],[1,-1,1,-1,1],[1,1,-1,-1,1]]:
                U = np.zeros((n, m))
                U[i, :] = pattern
                for k in range(p):
                    for vpat in [[1,1,1,1,1],[1,-1,1,-1,1]]:
                        V = np.zeros((m, p))
                        V[:, k] = vpat
                        W = np.zeros((p, n))
                        W[k, i] = 1
                        candidates.append((U, V, W))
        
        # Pattern 5: Kronecker-like structures
        for i in range(n):
            for j in range(m):
                U = np.zeros((n, m))
                U[i, j] = 1
                for k1 in range(p):
                    for k2 in range(k1+1, p):
                        for s in [1, -1]:
                            V = np.zeros((m, p))
                            V[j, k1] = 1
                            V[j, k2] = s
                            W = np.zeros((p, n))
                            W[k1, i] = 1
                            W[k2, i] = s
                            candidates.append((U, V, W))
        
        return candidates

    def compute_contribution(U, V, W):
        """Compute the 120 constraint values for a rank-1 term."""
        contrib = np.zeros(total_constraints)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    contrib[idx] = U[i, j] * V[j, k] * W[k, i]
        return contrib

    # Generate candidates
    candidates = generate_candidates()
    
    # Compute contribution vectors for all candidates
    print(f"Generated {len(candidates)} candidate terms", flush=True)
    
    contribs = []
    for idx, (U, V, W) in enumerate(candidates):
        contribs.append(compute_contribution(U, V, W))
        if time.time() - t0 > DEADLINE - 30:
            break
    
    # We need: sum_{c in selected} coeff_c * contrib_c = ones(120)
    # This is a system of 120 equations. Let's use exact rational arithmetic.
    
    # First, try greedy: pick candidates that cover many constraints
    target = np.ones(total_constraints)
    
    # Build a sparse representation
    # For each candidate, which constraints does it affect?
    candidate_supports = []
    for c in contribs:
        support = np.nonzero(c)[0]
        candidate_supports.append(support)
    
    # Greedy set cover approach with exact verification
    # Start with all 120 constraints uncovered
    uncovered = set(range(total_constraints))
    selected = []
    
    # Score each candidate by how many uncovered constraints it helps
    def score_candidate(ci):
        if ci >= len(contribs):
            return 0
        s = candidate_supports[ci]
        return len([x for x in s if x in uncovered])
    
    # Iterative greedy selection
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() - t0 > DEADLINE - 20:
            break
        if not uncovered:
            break
        
        # Find best candidate
        best_ci = -1
        best_score = 0
        for ci in range(len(contribs)):
            s = score_candidate(ci)
            if s > best_score:
                best_score = s
                best_ci = ci
        
        if best_ci < 0 or best_score == 0:
            break
        
        selected.append(best_ci)
        
        # Update uncovered: this candidate contributes to its support
        # But we need exact coefficients, so just track coverage
        for x in candidate_supports[best_ci]:
            uncovered.discard(x)
        
        if len(selected) >= 120:
            break
    
    print(f"Greedy selected {len(selected)} terms", flush=True)
    
    # Now solve for exact coefficients using rational arithmetic
    # We have len(selected) unknowns and 120 equations
    # If len(selected) > 120, we have an overdetermined system
    
    if len(selected) > 0:
        # Build the system: M * x = 1 where M is 120 x len(selected)
        # Use numpy for initial solve, then verify exactly
        M = np.zeros((total_constraints, len(selected)))
        for ci, idx in enumerate(selected):
            M[:, ci] = contribs[idx]
        
        # Least squares solution
        try:
            x, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
            
            # Check if solution is close to integer/half-integer
            # Round to nearest half-integer
            x_rounded = np.round(x * 2) / 2
            
            # Verify the rounded solution
            residual = M @ x_rounded - target
            max_err = np.max(np.abs(residual))
            
            if max_err < 1e-9:
                # Found a valid decomposition!
                rank = len(selected)
                print(f"Found valid decomposition with rank {rank}!", flush=True)
                
                # Build U, V, W matrices
                U = np.zeros((num_U, rank))
                V = np.zeros((num_V, rank))
                W = np.zeros((num_W, rank))
                
                for ci, idx in enumerate(selected):
                    coeff = x_rounded[ci]
                    if abs(coeff) < 1e-12:
                        continue
                    U_mat, V_mat, W_mat = candidates[idx]
                    U[:, ci] = U_mat.flatten() * coeff
                    V[:, ci] = V_mat.flatten() * coeff
                    W[:, ci] = W_mat.flatten() * coeff
                
                # Final verification
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(rank):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(val - 1) > 1e-9:
                                print(f"Verification failed at ({i},{j},{k}): {val}", flush=True)
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    # All passed
                    U_list = [[float(U[i, c]) for c in range(rank)] for i in range(num_U)]
                    V_list = [[float(V[i, c]) for c in range(rank)] for i in range(num_V)]
                    W_list = [[float(W[i, c]) for c in range(rank)] for i in range(num_W)]
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception as e:
            print(f"LSQ failed: {e}", flush=True)
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(num_U)]
    V = [[0] * rank for _ in range(num_V)]
    W = [[0] * rank for _ in range(num_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
