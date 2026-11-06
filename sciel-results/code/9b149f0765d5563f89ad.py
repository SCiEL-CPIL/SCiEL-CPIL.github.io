import time
import numpy as np
from scipy import sparse
from scipy.sparse import linalg

# Build the Hamiltonian components for a 6-site open XXZ Heisenberg chain
# Anisotropy gamma = 0.8, field h = 0.3
# A = odd bonds + half field, B = even bonds + half field

def _build_matrices():
    n = 6  # number of sites
    dim = 2 ** n
    gamma = 0.8
    h = 0.3
    
    # Spin-1/2 operators
    Sx = np.array([[0, 0.5], [0.5, 0]])
    Sy = np.array([[0, -0.5j], [0.5j, 0]])
    Sz = np.array([[0.5, 0], [0, -0.5]])
    
    # Build full Hamiltonian H = sum_{j=0}^{n-2} [Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + gamma Sz_j Sz_{j+1}] + h sum_j Sz_j
    # A = odd bonds (j=1,3) + half field
    # B = even bonds (j=0,2,4) + half field
    
    def apply_op(op, site, state):
        """Apply single-site operator op at given site to state vector."""
        # Reshape state to (2,)*n
        state = state.reshape([2] * n)
        # Apply op at site
        new_state = np.tensordot(op, state, axes=([1], [site]))
        # Move axes to correct position
        axes = list(range(n))
        del axes[site]
        new_state = new_state.transpose([site] + axes)
        return new_state.reshape(dim)
    
    def bond_hamiltonian(j, gamma):
        """Return the 2x2 bond Hamiltonian matrix for sites j, j+1 as sparse."""
        # H_bond = Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + gamma Sz_j Sz_{j+1}
        # In basis |00>, |01>, |10>, |11>:
        # Sx⊗Sx = 0.25 * [[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]
        # Sy⊗Sy = 0.25 * [[0,0,-1,0],[0,0,0,-1],[-1,0,0,0],[0,-1,0,0]]
        # Sz⊗Sz = 0.25 * [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
        M = np.array([
            [0, 0, 0.25 - 0.25*gamma, 0],
            [0, 0, 0, 0],
            [0.25 - 0.25*gamma, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        # Wait, let me recompute properly
        # Sx⊗Sx: |01><10| + |10><01| times 0.25
        # Sy⊗Sy: -|01><10| - |10><01| times 0.25  ... actually Sy⊗Sy gives -0.25*(|01><10|+|10><01|)
        # So Sx⊗Sx + Sy⊗Sy = 0.25*(|01><10|+|10><01|) - 0.25*(|01><10|+|10><01|) = 0? No!
        # Let me be more careful.
        # Sx = 0.5*[[0,1],[1,0]], Sy = 0.5*[[0,-i],[i,0]]
        # Sx⊗Sx = 0.25*[[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]
        # Sy⊗Sy = 0.25*[[0,0,-1,0],[0,0,0,-1],[-1,0,0,0],[0,-1,0,0]]
        # Sum = 0.25*[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]? No that's wrong.
        # Sx⊗Sx + Sy⊗Sy:
        # Element (0,2): 0.25 + (-0.25) = 0
        # Element (2,0): 0.25 + (-0.25) = 0
        # Hmm, so the XX+YY part only connects |00><11| and |11><00|? No...
        # Actually: Sx⊗Sx + Sy⊗Sy = 0.5*(S+⊗S- + S-⊗S+)
        # S+⊗S- connects |01> to |10>, S-⊗S+ connects |10> to |01>
        # So the matrix is:
        # [[0,0,0,0],[0,0,0.5,0],[0,0.5,0,0],[0,0,0,0]]
        # Wait: 0.5*(S+⊗S- + S-⊗S+)
        # S+⊗S-: maps |0,1> -> |1,0>, so element (2,1) = 1
        # S-⊗S+: maps |1,0> -> |0,1>, so element (1,2) = 1
        # So matrix has 0.5 at (1,2) and (2,1)
        
        M = np.array([
            [0, 0, 0, 0],
            [0, 0, 0.5, 0],
            [0, 0.5, 0, 0],
            [0, 0, 0, 0]
        ])
        # Sz⊗Sz = 0.25*diag(1, -1, -1, 1)
        M += gamma * 0.25 * np.diag([1, -1, -1, 1])
        return M
    
    # Build sparse matrices A and B
    # A: odd bonds j=1,3 plus half field
    # B: even bonds j=0,2,4 plus half field
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for j in range(n - 1):
        M_bond = bond_hamiltonian(j, gamma)
        # Apply to sites j, j+1
        for state_idx in range(dim):
            # Get the two-site state
            s1 = (state_idx >> (n - 1 - j)) & 1
            s2 = (state_idx >> (n - 1 - (j + 1))) & 1
            local_idx = s1 * 2 + s2
            for local_idx2 in range(4):
                val = M_bond[local_idx, local_idx2]
                if abs(val) > 1e-15:
                    s1_2 = local_idx2 // 2
                    s2_2 = local_idx2 % 2
                    if s1_2 != s1 or s2_2 != s2:
                        new_idx = state_idx
                        # Clear bits j and j+1
                        mask = (1 << (n - 1 - j)) | (1 << (n - 1 - (j + 1)))
                        new_idx = state_idx & ~mask
                        # Set new bits
                        new_idx |= (s1_2 << (n - 1 - j))
                        new_idx |= (s2_2 << (n - 1 - (j + 1)))
                        if j % 2 == 1:  # odd bond -> A
                            A[state_idx, new_idx] += val
                        else:  # even bond -> B
                            B[state_idx, new_idx] += val
    
    # Add field: h * sum_j Sz_j, split half to A, half to B
    # Sz_j is diagonal: +0.5 if spin up (bit=0), -0.5 if spin down (bit=1)
    for state_idx in range(dim):
        sz_sum = 0.0
        for j in range(n):
            bit = (state_idx >> (n - 1 - j)) & 1
            sz_sum += 0.5 if bit == 0 else -0.5
        A[state_idx, state_idx] += 0.5 * h * sz_sum
        B[state_idx, state_idx] += 0.5 * h * sz_sum
    
    return A, B

# Precompute matrices
_A, _B = _build_matrices()
_H = _A + _B

def _spectral_norm_error(a_list, b_list):
    """Compute mean spectral norm error at t = 0.25, 0.5, 1.0."""
    times = [0.25, 0.5, 1.0]
    dim = len(_A)
    errors = []
    
    # Precompute exp(-i*A) and exp(-i*B) using eigen-decomposition
    # Since A and B are Hermitian, we can use eigh
    evals_A, evecs_A = np.linalg.eigh(_A)
    evals_B, evecs_B = np.linalg.eigh(_B)
    exp_A = evecs_A @ np.diag(np.exp(-1j * evals_A)) @ evecs_A.conj().T
    exp_B = evecs_B @ np.diag(np.exp(-1j * evals_B)) @ evecs_B.conj().T
    
    # Precompute exp(-i*t*(A+B)) for each t
    evals_H, evecs_H = np.linalg.eigh(_H)
    
    for t in times:
        # Product formula: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Since A and B don't commute, order matters
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            a_k = a_list[k]
            b_k = b_list[k]
            # exp(-i a_k t A)
            if abs(a_k) > 1e-15:
                phase_A = np.exp(-1j * a_k * t * evals_A)
                exp_Ak = evecs_A @ np.diag(phase_A) @ evecs_A.conj().T
                U = exp_Ak @ U
            if abs(b_k) > 1e-15:
                phase_B = np.exp(-1j * b_k * t * evals_B)
                exp_Bk = evecs_B @ np.diag(phase_B) @ evecs_B.conj().T
                U = exp_Bk @ U
        
        # Exact propagator
        phase_H = np.exp(-1j * t * evals_H)
        U_exact = evecs_H @ np.diag(phase_H) @ evecs_H.conj().T
        
        # Spectral norm of difference
        diff = U - U_exact
        # Use SVD for spectral norm
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0])
    
    return np.mean(errors)

def coefficients(stages):
    """Optimized product formula coefficients using gradient descent with symmetry constraints."""
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin within 150s limit
    
    # Use symmetry: a and b are palindromic (a_k = a_{13-k}, b_k = b_{13-k})
    # This reduces parameters from 24 to 12
    # a = [a0, a1, a2, a3, a4, a5, a5, a4, a3, a2, a1, a0]
    # b = [b0, b1, b2, b3, b4, b5, b5, b4, b3, b2, b1, b0]
    
    # Parameters: x = [a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5]
    # Constraints: sum(a) = 1, sum(b) = 1
    # sum(a) = 2*(a0+a1+a2+a3+a4+a5) = 1
    # sum(b) = 2*(b0+b1+b2+b3+b4+b5) = 1
    
    def unpack(x):
        a = [x[0], x[1], x[2], x[3], x[4], x[5], x[5], x[4], x[3], x[2], x[1], x[0]]
        b = [x[6], x[7], x[8], x[9], x[10], x[11], x[11], x[10], x[9], x[8], x[7], x[6]]
        return a, b
    
    def objective(x):
        a, b = unpack(x)
        return _spectral_norm_error(a, b)
    
    # Initial guess: symmetric Strang-like
    # a_k = 1/12 for all k, b_k = 1/12 for all k
    # But with palindrome: 2*sum(a0..a5) = 1 => sum(a0..a5) = 1/2
    x0 = np.array([1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12])
    
    # Check constraints
    a0, b0 = unpack(x0)
    sum_a0 = sum(a0)
    sum_b0 = sum(b0)
    print(f"Initial: sum_a={sum_a0:.6f}, sum_b={sum_b0:.6f}")
    
    best_x = x0.copy()
    best_val = objective(best_x)
    print(f"Initial objective: {best_val:.8e}")
    
    # Gradient descent with numerical gradients
    lr = 0.01
    n_iter = 0
    
    while time.time() < deadline - 5:  # 5s safety margin
        n_iter += 1
        # Compute gradient numerically
        grad = np.zeros(12)
        f0 = objective(best_x)
        eps = 1e-6
        for i in range(12):
            x_plus = best_x.copy()
            x_plus[i] += eps
            f_plus = objective(x_plus)
            grad[i] = (f_plus - f0) / eps
        
        # Update
        new_x = best_x - lr * grad
        
        # Project onto constraint manifold:
        # sum(a) = 2*sum(x[0:6]) = 1
        # sum(b) = 2*sum(x[6:12]) = 1
        sum_a_half = np.sum(new_x[0:6])
        sum_b_half = np.sum(new_x[6:12])
        if abs(sum_a_half) > 1e-15:
            new_x[0:6] = new_x[0:6] / sum_a_half * 0.5
        if abs(sum_b_half) > 1e-15:
            new_x[6:12] = new_x[6:12] / sum_b_half * 0.5
        
        # Clip to [-50, 50]
        new_x = np.clip(new_x, -50, 50)
        
        new_val = objective(new_x)
        
        if new_val < best_val:
            best_x = new_x
            best_val = new_val
            # Adaptive learning rate
            if n_iter % 50 == 0:
                lr = max(lr * 0.95, 1e-5)
        else:
            # Reduce learning rate
            lr *= 0.5
        
        if n_iter % 20 == 0:
            print(f"Iter {n_iter}: val={best_val:.8e}, lr={lr:.6e}")
    
    # Final polish with smaller steps
    remaining = deadline - time.time()
    if remaining > 3:
        lr2 = 1e-4
        for _ in range(int(min(remaining / 0.5, 200))):  # at most ~200 iters or until time
            if time.time() > deadline - 2:
                break
            grad = np.zeros(12)
            f0 = objective(best_x)
            eps = 1e-7
            for i in range(12):
                x_plus = best_x.copy()
                x_plus[i] += eps
                f_plus = objective(x_plus)
                grad[i] = (f_plus - f0) / eps
            new_x = best_x - lr2 * grad
            sum_a_half = np.sum(new_x[0:6])
            sum_b_half = np.sum(new_x[6:12])
            if abs(sum_a_half) > 1e-15:
                new_x[0:6] = new_x[0:6] / sum_a_half * 0.5
            if abs(sum_b_half) > 1e-15:
                new_x[6:12] = new_x[6:12] / sum_b_half * 0.5
            new_x = np.clip(new_x, -50, 50)
            new_val = objective(new_x)
            if new_val < best_val:
                best_x = new_x
                best_val = new_val
    
    # Construct final a, b lists
    a_final = np.array([best_x[0], best_x[1], best_x[2], best_x[3], best_x[4], best_x[5],
                        best_x[5], best_x[4], best_x[3], best_x[2], best_x[1], best_x[0]])
    b_final = np.array([best_x[6], best_x[7], best_x[8], best_x[9], best_x[10], best_x[11],
                        best_x[11], best_x[10], best_x[9], best_x[8], best_x[7], best_x[6]])
    
    # Ensure sums are exactly 1 (numerical drift correction)
    sum_a = np.sum(a_final)
    sum_b = np.sum(b_final)
    
    if abs(sum_a) > 1e-15:
        a_final = a_final / sum_a
    if abs(sum_b) > 1e-15:
        b_final = b_final / sum_b
    
    # Final clip
    a_final = np.clip(a_final, -50, 50)
    b_final = np.clip(b_final, -50, 50)
    
    # Re-check sums after clipping
    sum_a = np.sum(a_final)
    sum_b = np.sum(b_final)
    if abs(sum_a) > 1e-15:
        a_final = a_final / sum_a
    if abs(sum_b) > 1e-15:
        b_final = b_final / sum_b
    
    return a_final.tolist(), b_final.tolist()
