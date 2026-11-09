"""Combined program: structured block/Kronecker search + gradient refinement."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p

    # ---- Schoolbook baseline (guaranteed valid) ----
    U = np.zeros((nm, rank_school))
    V = np.zeros((mp, rank_school))
    W = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school

    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    # Support: (i_idx, j_idx, k_idx) where tensor value is 1
    support = [(a * m + b, b * p + c, c * n + a)
               for a in range(n) for b in range(m) for c in range(p)]

    def matmul_tensor(Um, Vm, Wm):
        """Compute the full n*m*p tensor from factor matrices."""
        T = np.zeros((n, m, p))
        for r in range(Um.shape[1]):
            u = Um[:, r]
            v = Vm[:, r]
            w = Wm[:, r]
            A = u.reshape(n, m)
            B = v.reshape(m, p)
            C = w.reshape(p, n).T  # C[i, a]
            T += np.einsum('ij,jk->ik', A, B) * C
        return T

    def validate_full(Um, Vm, Wm, tol=1e-8):
        T = matmul_tensor(Um, Vm, Wm)
        target = np.zeros((n, m, p))
        for (i_idx, j_idx, k_idx) in support:
            a, b = divmod(i_idx, m)
            b2, c = divmod(j_idx, p)
            target[a, b, c] = 1.0
        return np.allclose(T, target, atol=tol)

    def to_frac(x, max_den=1000):
        """Snap a float to a small rational; return None if too far."""
        if abs(x) < 1e-12:
            return Fraction(0)
        f = Fraction(x).limit_denominator(max_den)
        if abs(float(f) - x) < 1e-6:
            return f
        return None

    def snap_halfinteger(Um, Vm, Wm):
        """Try to snap all entries to half-integers and revalidate."""
        U2 = Um.copy(); V2 = Vm.copy(); W2 = Wm.copy()
        ok = True
        for M in (U2, V2, W2):
            for r in range(M.shape[0]):
                for c in range(M.shape[1]):
                    f = to_frac(M[r, c])
                    if f is None:
                        ok = False
                        break
                    M[r, c] = float(f)
                if not ok:
                    break
            if not ok:
                break
        if ok and validate_full(U2, V2, W2, tol=1e-7):
            return U2, V2, W2, True
        return None, None, None, False

    # =========================================================
    # STRUCTURAL FAMILY: block-diagonal / Kronecker decomposition
    # =========================================================
    # Group the m shared indices into blocks. Each block b has size m_b and
    # scalar lambda_b. Block contribution to C[i,a]:
    #   lambda_b * (sum_{j in block} A[i,j]) * (sum_{k in block} B[k,a])
    # We need sum_b lambda_b * S_b[i] * T_b[a] = delta_{i,a} for all i,a.
    # For a fixed (i,a) with i != a this must vanish; for i == a it must be 1.
    # With blocks this becomes a small linear system over the block scalars
    # (and we can also allow per-block row/col sign patterns).

    def try_block_partition(partition, time_left):
        """partition: list of tuples of shared-index positions (each a subset of range(m)).
        Build a candidate and validate. Returns (U,V,W,rank) or None."""
        # For each block we choose a scalar lambda. The block's row vector over
        # A is the indicator of the block (summed), similarly for B.
        # Block b produces: lambda_b * (sum_{j in block} A[i,j]) * (sum_{k in block} B[k,a])
        # Define for each (i,a): R_b(i) = number of block members... actually
        # sum_{j in block} A[i,j] is a full linear form, not a scalar per i.
        # So the block tensor is: lambda_b * u_b (n*m) outer v_b (m*p) outer w_b (p*n)
        # where u_b[i*m+j] = 1 if j in block else 0
        #        v_b[j*p+k] = 1 if k in block else 0
        #        w_b[k*n+i] = 1 (for all) ... no, w must couple.
        # Actually a single rank-1 term with u_b, v_b, w_b gives
        #   lambda_b * (sum_{j in block} A[i,j]) * (sum_{k in block} B[k,a]) * w_b[k*n+i]
        # For this to be a clean "outer product of row-sum and col-sum" we need
        # w_b to be constant 1 over all (k,i). Then block term =
        #   lambda_b * (sum_{j in block} A[i,j]) * (sum_{k in block} B[k,a])
        # which is independent of k in w. Good. So w_b is all-ones.
        n_blocks = len(partition)
        U = np.zeros((nm, n_blocks))
        V = np.zeros((mp, n_blocks))
        W = np.zeros((pn, n_blocks))
        for b, block in enumerate(partition):
            for j in block:
                for i in range(n):
                    U[i * m + j, b] = 1.0
                for k in block:
                    for a in range(p):
                        V[k * p + a, b] = 1.0
            for k in range(p):
                for i in range(n):
                    W[k * n + i, b] = 1.0
        # Now solve for lambda_b (n_blocks unknowns) such that
        # sum_b lambda_b * (rowsum_b(i)) * (colsum_b(a)) = delta_{i,a}
        # rowsum_b(i) = sum_{j in block_b} A[i,j]  -> but A is unknown!
        # This is the crux: rowsum depends on A. So a pure block-sum with
        # all-ones w does NOT give a fixed tensor independent of A.
        # => This family only works if we instead make each block a full
        #    identity-like sub-multiplication. That requires m_b = m (trivial).
        # Therefore the simple block-sum family is infeasible for m>1 blocks.
        return None

    # The above reveals the block-sum family is infeasible. Instead, use a
    # genuinely structural family: Kronecker / tensor-product decomposition.
    # Write A (n x m) and B (m x p) as products when m is composite.
    # m = 5 is prime, so m cannot be factored. But we can still try a
    # "diagonal embedding": treat the m=5 shared dimension as a sum of 5
    # independent 1x1 multiplications (schoolbook) and try to merge pairs
    # using the fact that C[i,a] only needs the total sum.

    # ------------------------------------------------------------------
    # FINE-GRAINED LOCAL SEARCH (structural refinement): start from
    # schoolbook and try to MERGE two terms into one by solving for a
    # replacement rank-1 term that reproduces both original terms'
    # contributions on the support while cancelling off-support.
    # This is a different search family than blind gradient descent.
    # ------------------------------------------------------------------
    def merge_two_terms(Um, Vm, Wm, r1, r2, time_left):
        """Try to replace terms r1 and r2 with a single term."""
        # The two terms contribute to each support triple (i,j,k):
        #   t1 = U[i,r1]*V[j,r1]*W[k,r1], t2 = U[i,r2]*V[j,r2]*W[k,r2]
        # We want a single term (u,v,w) such that for all support triples:
        #   u[i]*v[j]*w[k] = t1 + t2
        # This is a rank-1 fit to the 2-term sum. In general the sum of two
        # rank-1 tensors is rank 2, so a single rank-1 term cannot reproduce
        # it unless the two terms are proportional. So merging is only
        # possible if term r1 and r2 are proportional on the support.
        # Check proportionality: does there exist scalar c with
        #   U[:,r1]*V[:,r1]*W[:,r1] ~ c * U[:,r2]*V[:,r2]*W[:,r2]?
        # Equivalent: the two rank-1 tensors are scalar multiples.
        # Test on a few support points.
        # Build the two "tensor" vectors over support.
        v1 = np.array([Um[i, r1] * Vm[j, r1] * Wm[k, r1] for (i, j, k) in support])
        v2 = np.array([Um[i, r2] * Vm[j, r2] * Wm[k, r2] for (i, j, k) in support])
        # If v1 and v2 are proportional, we can merge.
        # Use least squares to find c minimizing ||v1 - c v2||.
        if np.linalg.norm(v2) < 1e-12:
            return None
        c = np.dot(v1, v2) / np.dot(v2, v2)
        resid = np.linalg.norm(v1 - c * v2)
        scale = np.linalg.norm(v1)
        if resid < 1e-9 * max(1.0, scale):
            # Proportional: replace r2 with r1*c (absorb) and drop r2.
            # New term r1' = r1 + c*r2 component. Actually just add.
            Um2 = Um.copy(); Vm2 = Vm.copy(); Wm2 = Wm.copy()
            # Merge: term r1 becomes r1 + c*r2 (in factor space this is not
            # simply additive because it's a tensor, but since they're
            # proportional rank-1 tensors, the sum is (u1+c u2, v1, w1) etc.
            # We need the actual factor vectors. Since t1 = c t2 pointwise on
            # support, the factors are proportional: u1 = a u2, v1 = b v2,
            # w1 = d w2 with a*b*d = c. We can just keep both factors and
            # scale: use u1+v1, v2, w2? That changes the tensor.
            # Safest: keep the merged term as the sum by adding the two
            # rank-1 terms is NOT a single rank-1 term. So merging two
            # proportional rank-1 terms into one rank-1 term:
            # (u1,v1,w1) and (u2,v2,w2) with (u1,v1,w1)=c(u2,v2,w2) means
            # u1=c u2 etc only if all three proportional by same c. That's
            # not guaranteed. So merging is generally infeasible.
            return None
        return None

    # Merge family is infeasible in general. Proceed to the most promising
    # structural family: randomized rank-r initialization + accelerated
    # gradient descent with support-only loss and off-support penalty,
    # sweeping r from a low guess upward. This is the refinement machinery
    # from the current program, but we add: (a) a better structured init
    # using the Kronecker idea for the (i,a) identity, (b) L-BFGS-like
    # momentum, and (c) an off-support penalty term so the loss is smooth.

    def loss_and_grad(Um, Vm, Wm, off_weight=1.0):
        """Smooth loss: sum over support (recon-1)^2 + off_weight * sum over
        sampled off-support recon^2. Returns loss and grads."""
        L = 0.0
        dU = np.zeros_like(Um); dV = np.zeros_like(Vm); dW = np.zeros_like(Wm)
        # Support: must equal 1
        for (i, j, k) in support:
            recon = np.dot(Um[i, :] * Vm[j, :], Wm[k, :])
            e = recon - 1.0
            L += e * e
            dU[i, :] += 2 * e * Vm[j, :] * Wm[k, :]
            dV[j, :] += 2 * e * Um[i, :] * Wm[k, :]
            dW[k, :] += 2 * e * Um[i, :] * Vm[j, :]
        # Off-support: sample more entries for better gradient signal
        if off_weight > 0:
            rng = np.random.RandomState(42)
            n_samples = 2000
            for _ in range(n_samples):
                ii = rng.randint(nm); jj = rng.randint(mp); kk = rng.randint(pn)
                a, b = divmod(ii, m)
                b2, c = divmod(jj, p)
                c2, a2 = divmod(kk, n)
                if b == b2 and c == c2 and a == a2:
                    continue  # this is support, skip
                recon = np.dot(Um[ii, :] * Vm[jj, :], Wm[kk, :])
                L += off_weight * recon * recon
                dU[ii, :] += 2 * off_weight * recon * Vm[jj, :] * Wm[kk, :]
                dV[jj, :] += 2 * off_weight * recon * Um[ii, :] * Wm[kk, :]
                dW[kk, :] += 2 * off_weight * recon * Um[ii, :] * Vm[jj, :]
        return L, dU, dV, dW

    def gradient_descent(U0, V0, W0, steps, lr, time_left):
        U = U0.copy(); V = V0.copy(); W = W0.copy()
        best = (float('inf'), U.copy(), V.copy(), W.copy())
        # momentum
        mU = np.zeros_like(U); mV = np.zeros_like(V); mW = np.zeros_like(W)
        beta = 0.9
        for s in range(steps):
            if time.time() > time_left:
                break
            L, dU, dV, dW = loss_and_grad(U, V, W)
            if L < best[0]:
                best = (L, U.copy(), V.copy(), W.copy())
            mU = beta * mU + (1 - beta) * dU
            mV = beta * mV + (1 - beta) * dV
            mW = beta * mW + (1 - beta) * dW
            U -= lr * mU
            V -= lr * mV
            W -= lr * mW
            # Re-normalize each rank-1 term to control scale drift
            # For each column t, scale U[:,t] by 1/sqrt(norm) and W[:,t] by sqrt(norm)
            for t in range(U.shape[1]):
                nU = np.linalg.norm(U[:, t])
                nW = np.linalg.norm(W[:, t])
                if nU > 1e-12 and nW > 1e-12:
                    scale = np.sqrt(nW / nU)
                    U[:, t] *= scale
                    W[:, t] /= scale
            if s % 50 == 0 and s > 0:
                lr *= 0.92
        return best

    # Structured init: for target rank r, initialize each term to help build
    # the identity C[i,a] = delta_{i,a}. Each term picks a random (i0, a0)
    # pair and a random shared-index vector v_j. The term contributes
    # v_j * v_j * w to C[i0, a0] (diagonal if i0==a0, off-diagonal otherwise).
    # We set w to normalize the diagonal contribution to ~1/r so that r terms
    # sum to ~1 on the diagonal.
    def structured_init(r, seed):
        rng = np.random.RandomState(seed)
        U = np.zeros((nm, r)); V = np.zeros((mp, r)); W = np.zeros((pn, r))
        # Assign terms to diagonal pairs first (n*p = 18 diagonal entries)
        diag_pairs = [(i, i) for i in range(min(n, p))]
        offdiag_pairs = [(i, a) for i in range(n) for a in range(p) if i != a]
        all_pairs = diag_pairs + offdiag_pairs
        for t in range(r):
            if t < len(all_pairs):
                i0, a0 = all_pairs[t]
            else:
                i0 = rng.randint(n); a0 = rng.randint(p)
            # Random shared-index vector with non-zero norm
            jpat = rng.randn(m)
            norm_sq = np.dot(jpat, jpat)
            if norm_sq < 1e-10:
                jpat = np.ones(m) / np.sqrt(m)
                norm_sq = 1.0
            # Scale so that the diagonal contribution is ~1/r
            # Term contributes (sum_j U[i0,j]*V[j,a0]) * W[a0*n+i0, t] to C[i0,a0]
            # If i0==a0: sum_j jpat[j]*jpat[j] = norm_sq
            # We want norm_sq * w = 1/r, so w = 1/(r*norm_sq)
            w_val = 1.0 / (r * norm_sq)
            for j in range(m):
                U[i0 * m + j, t] = jpat[j]
                V[j * p + a0, t] = jpat[j]
            W[a0 * n + i0, t] = w_val
        return U, V, W

    # Sweep ranks. Start low and go up; stop at first valid full solution.
    min_rank = max(12, n * p)  # theoretical lower bound ~ n*p for square-ish
    max_rank = rank_school - 1
    for target_rank in range(min_rank, max_rank + 1):
        if time.time() > deadline - 25:
            break
        # A few random structured inits
        n_init = max(1, min(4, int((deadline - time.time()) / 15)))
        for init_idx in range(n_init):
            if time.time() > deadline - 20:
                break
            U0, V0, W0 = structured_init(target_rank, seed=1000 + init_idx * 7 + target_rank)
            L, Uf, Vf, Wf = gradient_descent(U0, V0, W0, steps=600, lr=0.005,
                                             time_left=deadline - 20)
            # Check support max error
            max_err = 0.0
            for (i, j, k) in support:
                recon = np.dot(Uf[i, :] * Vf[j, :], Wf[k, :])
                max_err = max(max_err, abs(recon - 1.0))
            if max_err < 1e-6:
                # Full validation
                if validate_full(Uf, Vf, Wf, tol=1e-7):
                    # Try to snap to half-integers
                    Us, Vs, Ws, snapped = snap_halfinteger(Uf, Vf, Wf)
                    if snapped:
                        best_U, best_V, best_W = Us, Vs, Ws
                    else:
                        best_U, best_V, best_W = Uf, Vf, Wf
                    best_rank = target_rank
                    break
        if best_rank < rank_school:
            break

    # Final safety: if we have a sub-schoolbook solution, ensure it's valid;
    # otherwise fall back to the guaranteed schoolbook.
    if best_rank < rank_school:
        if not validate_full(best_U, best_V, best_W, tol=1e-7):
            best_U, best_V, best_W = U, V, W
            best_rank = rank_school

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
