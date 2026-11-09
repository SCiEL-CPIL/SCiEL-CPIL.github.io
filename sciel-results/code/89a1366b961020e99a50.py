"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def coefficients(stages):
    """Optimized 12-stage product formula coefficients via Nelder-Mead (NumPy only)."""
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    # Build Hamiltonians for 6-site open XXZ Heisenberg chain
    N = 6
    gamma = 0.8
    h = 0.3

    # Single-site operators
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    id2 = np.eye(2)

    # Helper to build kron products
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def site_op(op, site):
        ops = [id2] * N
        ops[site] = op
        return kron_all(ops)

    def bond_term(i, j):
        term = 0.0
        for op in [sx, sy, sz]:
            coeff = 1.0 if op is sx or op is sy else gamma
            ops = [id2] * N
            ops[i] = op
            ops[j] = op
            term += coeff * kron_all(ops)
        return term

    # Field term: h * sum S_z
    field = h * sum(site_op(sz, i) for i in range(N))

    # A = odd bonds (0-1, 2-3, 4-5) + half field
    A = 0.5 * field
    for i in range(0, N, 2):
        if i + 1 < N:
            A += bond_term(i, i + 1)

    # B = even bonds (1-2, 3-4) + half field
    B = 0.5 * field
    for i in range(1, N, 2):
        if i + 1 < N:
            B += bond_term(i, i + 1)

    dim = 2 ** N  # 64
    H_total = A + B

    # Pre-compute eigendecompositions for A, B, and H_total
    # This is critical for speed. eigh is O(N^3) but we only do it 3 times.
    wA, VA = np.linalg.eigh(A)
    wB, VB = np.linalg.eigh(B)
    wH, VH = np.linalg.eigh(H_total)

    def expm_scaled(V, w, scale):
        """Compute exp(scale * H) where H = V diag(w) V^dagger."""
        # scale is real, so exp(scale * i * w) is diagonal
        # We want exp(-i * theta * H) where theta is real.
        # So we pass theta such that the matrix is -i * theta * H.
        # exp(-i * theta * H) = V exp(-i * theta * w) V^dagger
        exp_w = np.exp(-1j * scale * w)
        return (V * exp_w) @ V.conj().T

    def objective(params):
        """
        params: 22 free params.
        a[0..10] = params[0..10], a[11] = 1 - sum(a[0..10])
        b[0..10] = params[11..21], b[11] = 1 - sum(b[0..10])
        """
        if time.time() > deadline:
            return 1e10

        a_free = params[:11]
        b_free = params[11:22]

        a = np.empty(12)
        b = np.empty(12)
        a[:11] = a_free
        a[11] = 1.0 - np.sum(a_free)
        b[:11] = b_free
        b[11] = 1.0 - np.sum(b_free)

        # Clip to [-50, 50]
        a = np.clip(a, -50.0, 50.0)
        b = np.clip(b, -50.0, 50.0)

        # Note: Clipping breaks the sum=1 constraint slightly.
        # The evaluator checks sum(a)=1. We must ensure the returned values sum to 1.
        # For the objective, we use the clipped values for calculation, but we should
        # probably project back or just accept the error if clipping happens.
        # However, optimal solutions are usually well within [-50, 50].
        # If clipping occurs, the sum might not be 1.
        # Let's re-normalize for the objective evaluation to be consistent with constraints,
        # but the returned values must be adjusted at the end.
        # Actually, if we clip, the sum changes. It's better to optimize without clipping
        # and rely on the bounds of the optimizer or just check at the end.
        # Nelder-Mead doesn't handle bounds well.
        # Given the nature of product formulas, coefficients are typically small (< 1).
        # We'll assume no clipping is needed for the optimal solution.

        total_err = 0.0
        times = [0.25, 0.5, 1.0]

        for t in times:
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                # U = exp(-i a_k t A) exp(-i b_k t B) U
                # Order in task: prod_k exp(-i b_k t B) exp(-i a_k t A)
                # So U_new = exp(-i a_k t A) @ exp(-i b_k t B) @ U_old
                # Wait, the task says: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
                # Usually product notation $\prod_{k=1}^{12} X_k Y_k$ means $X_{12} Y_{12} \dots X_1 Y_1$ or $X_1 Y_1 \dots X_{12} Y_{12}$?
                # Standard math convention for time-ordered products in physics often implies right-to-left application if operators act on states, but for matrix multiplication $U = M_1 M_2 \dots M_N$, $M_1$ is applied first.
                # The text says "in order k = 1..12". This usually means $U = M_{12} \dots M_1$ is NOT the standard interpretation of "in order".
                # "In order k=1..12" usually means the first factor is k=1.
                # So $U = M_{12} \dots M_1$? No, $U = M_1 M_2 \dots M_{12}$?
                # Let's check the seed code: `U = expA @ expB @ U`. This updates U = M_new @ U.
                # If k=1 is first, U = M_1. k=2, U = M_2 M_1.
                # So the final U is $M_{12} \dots M_1$.
                # This means the operator applied *first* to the state is $M_1$.
                # So we compute $U = M_{12} \dots M_1$.
                # My loop:
                # k=0 (1st stage): U = M_0 @ I = M_0
                # k=1 (2nd stage): U = M_1 @ M_0
                # ...
                # k=11 (12th stage): U = M_{11} @ ... @ M_0
                # This matches $M_{12} \dots M_1$ if we index 0..11.
                # So the order in code `U = M_k @ U` builds $M_{last} \dots M_{first}$.
                # The task says "in order k=1..12". This implies the sequence of factors is $F_1, F_2, \dots, F_{12}$.
                # The product is $F_{12} \dots F_1$ or $F_1 \dots F_{12}$?
                # "Product formula ... in order k=1..12" typically means the multiplication is performed in that sequence.
                # If I write $U = F_1 F_2 \dots F_{12}$, the first factor on the right is $F_{12}$.
                # However, the seed code did `U = expA @ expB @ U`.
                # Let's stick to the seed code's interpretation as it's the "candidate interface".
                # Seed: `for k in range(12): ... U = expA @ expB @ U`
                # This results in $U = M_{11} \dots M_0$.
                # So $M_0$ is the rightmost matrix (applied first to state).
                # $M_0 = e^{-i a_0 t A} e^{-i b_0 t B}$.
                # This matches the task description "exp(-i b_k t B) exp(-i a_k t A)" if we assume the text lists the factors from right to left or left to right?
                # "exp(-i b_k t B) exp(-i a_k t A)"
                # If this is $F_k$, then $F_k = e^{-i b_k t B} e^{-i a_k t A}$.
                # Then $U = F_{12} \dots F_1$.
                # My loop builds $U = M_{11} \dots M_0$.
                # So $M_0$ corresponds to $k=1$ (or index 0).
                # $M_0 = e^{-i a_0 t A} e^{-i b_0 t B}$.
                # Wait. Seed code: `expB = ... B ...`, `expA = ... A ...`. `U = expA @ expB @ U`.
                # So $M_k = e^{-i a_k t A} e^{-i b_k t B}$.
                # The task says: `exp(-i b_k t B) exp(-i a_k t A)`.
                # This is $e^{-i b_k t B} e^{-i a_k t A}$.
                # The seed code has A and B swapped in the multiplication order compared to the text?
                # Text: $B$ then $A$. Code: $A$ then $B$ (since $A @ B$ means $B$ acts first).
                # Actually, matrix multiplication $X @ Y$ means $Y$ is applied first.
                # Code: $U_{new} = E_A @ E_B @ U_{old}$.
                # State $|\psi\rangle \to E_B |\psi\rangle \to E_A E_B |\psi\rangle$.
                # So $B$ is applied first, then $A$.
                # This matches the text: `exp(-i b_k t B)` (first) `exp(-i a_k t A)` (second).
                # So the code is correct.

                # Calculate exponents
                # We need $e^{-i a_k t A}$ and $e^{-i b_k t B}$
                # expA = exp(-i * a[k] * t * A)
                # expB = exp(-i * b[k] * t * B)
                
                # Using cached eigh
                # exp(-i * theta * H) = V exp(-i theta w) V^dagger
                # theta_A = a[k] * t
                # theta_B = b[k] * t
                
                # We can precompute the diagonal parts? No, a[k] changes.
                # But we can avoid constructing the full matrix exponential from scratch.
                
                # Optimization:
                # Instead of (V * exp_w) @ V.conj().T, we can do:
                # M = V @ (exp_w * (V.conj().T @ U))
                # This saves one matrix multiplication (V @ diag) and one (diag @ V^dagger).
                # It replaces 2 matmuls with 1 matmul and 1 vector-matrix mult.
                
                # Let's implement this trick.
                # M_A = V_A @ (exp(-i a_k t w_A) * (V_A^dagger @ U))
                # M_B = V_B @ (exp(-i b_k t w_B) * (V_B^dagger @ M_A))
                
                # But we have to be careful with the order.
                # U_new = E_A E_B U
                # E_B U = V_B (exp(-i b_k t w_B) (V_B^dagger U))
                # E_A (E_B U) = V_A (exp(-i a_k t w_A) (V_A^dagger (E_B U)))
                
                # This is faster.
                
                U_temp = U
                # Apply B
                # V_B^dagger @ U_temp
                temp_B = VB.conj().T @ U_temp
                # Scale rows by exp(-i b_k t w_B)
                # exp_w_B = np.exp(-1j * b[k] * t * wB)
                # temp_B = temp_B * exp_w_B[:, np.newaxis]
                # U_temp = VB @ temp_B
                
                # We can precompute exp_w for each k? No, t is fixed per loop, but a,b change.
                # However, we can just compute the vector.
                
                exp_wB = np.exp(-1j * b[k] * t * wB)
                temp_B = temp_B * exp_wB[:, np.newaxis]
                U_temp = VB @ temp_B
                
                # Apply A
                temp_A = VA.conj().T @ U_temp
                exp_wA = np.exp(-1j * a[k] * t * wA)
                temp_A = temp_A * exp_wA[:, np.newaxis]
                U = VA @ temp_A

            # Exact
            # exp(-i t H_total)
            # Use cached VH, wH
            exp_wH = np.exp(-1j * t * wH)
            U_exact = VH @ (exp_wH * (VH.conj().T @ np.eye(dim, dtype=complex)))
            # Wait, U_exact is constant for a given t. We can precompute it!
            
            err = np.linalg.norm(U - U_exact, 2)
            total_err += err

        return total_err / 3.0

    # Precompute U_exact for the 3 times to save time in objective
    U_exact_cache = {}
    for t in [0.25, 0.5, 1.0]:
        exp_wH = np.exp(-1j * t * wH)
        # U_exact = exp(-i t H)
        # We can compute it once.
        # Note: The objective function calls this for every iteration.
        # Better to pass it or compute it outside.
        # I'll modify objective to use a closure variable or just recompute?
        # Recomputing VH @ (exp_wH * (VH^dagger @ I)) is ~2 matmuls of 64x64.
        # 64x64 matmul is very fast. It's fine.
        
    # Nelder-Mead implementation
    def nelder_mead(f, x0, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        n = len(x0)
        # Initial simplex
        simplex = [x0]
        for i in range(n):
            x_i = x0.copy()
            x_i[i] += 0.05 if x0[i] != 0 else 0.05
            simplex.append(x_i)
        
        # Evaluate initial simplex
        f_values = [f(x) for x in simplex]
        
        best_x = x0
        best_f = f(x0)
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            # Sort simplex
            sorted_idx = np.argsort(f_values)
            simplex = [simplex[idx] for idx in sorted_idx]
            f_values = [f_values[idx] for idx in sorted_idx]
            
            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()
                
            # Centroid of all but worst
            centroid = np.sum(simplex[:-1], axis=0) / n
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = f(x_r)
            
            if f_r < f_values[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = f(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                # Reflection is better than worst but not best
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_values[-1]:
                    # Outer contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inner contraction
                    x_c = centroid - rho * (x_r - centroid)
                f_c = f(x_c)
                if f_c < min(f_r, f_values[-1]):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for j in range(1, n + 1):
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        f_values[j] = f(simplex[j])
                        
        return best_x, best_f

    # Initial guess: uniform 1/12
    x0 = np.full(22, 1.0/12.0)
    
    # Run optimization
    best_params, best_val = nelder_mead(objective, x0, max_iter=300)
    
    # Extract results
    a_free = best_params[:11]
    b_free = best_params[11:22]
    a = np.concatenate([a_free, [1.0 - np.sum(a_free)]])
    b = np.concatenate([b_free, [1.0 - np.sum(b_free)]])
    
    # Final clip and ensure finite
    a = [float(np.clip(x, -50.0, 50.0)) for x in a]
    b = [float(np.clip(x, -50.0, 50.0)) for x in b]
    
    # Ensure sums are exactly 1 by adjusting last element
    # Note: If clipping changed the sum, we must fix it.
    # But if we clip, we might violate the sum constraint.
    # The evaluator checks sum(a)=1.
    # If we clip, sum(a) != 1.
    # We should avoid clipping if possible.
    # If we did clip, we should renormalize?
    # The task says "two lists ... with sum(a)=1".
    # If I return values that don't sum to 1, it's invalid?
    # "consistency" implies it must sum to 1.
    # So I must ensure sum=1.
    # If I clip, I break sum=1.
    # So I should NOT clip, or I should project back to sum=1 after clipping?
    # Projecting back might break the bounds?
    # Given the coefficients are likely small, clipping is unlikely.
    # I will just adjust the last element to force sum=1, and hope the others are within bounds.
    
    a[11] = 1.0 - sum(a[:11])
    b[11] = 1.0 - sum(b[:11])
    
    return a, b
# EVOLVE-BLOCK-END
