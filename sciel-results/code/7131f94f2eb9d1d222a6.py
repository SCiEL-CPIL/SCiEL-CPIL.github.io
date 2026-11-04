"""Matrix multiplication tensor decomposition for M_{3,4,7}."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Trivial rank
    trivial_rank = n * m * p

    # Build the trivial decomposition
    def build_trivial():
        U = np.zeros((n * m, trivial_rank), dtype=np.float64)
        V = np.zeros((m * p, trivial_rank), dtype=np.float64)
        W = np.zeros((p * n, trivial_rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U, V, W = build_trivial()

    # Try to find a lower-rank decomposition using a randomized algebraic approach.
    # Strategy: We look for a decomposition of rank r < trivial_rank.
    # The tensor T_{i,j,k} = 1 for all i,j,k (since matrix multiplication tensor
    # has T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'} but in our indexing
    # the "multiplication tensor" has T[i,j,k] = 1 when the indices match the
    # multiplication pattern).
    #
    # Actually, let me reconsider the tensor. The matrix multiplication tensor
    # M_{n,m,p} has entries T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p).
    # The tensor is defined such that for matrices A (n x m) and B (m x p),
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor decomposition satisfies:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'}*... 
    # Wait, let me re-read the problem.
    #
    # The problem says: "exactly decompose the matrix multiplication tensor"
    # and "W[k*n+i,r]" is the output factor.
    #
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k
    # if we think of it as: the (i,k)-th entry of the product is sum_j A[i,j]*B[j,k].
    # So the tensor is actually T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor
    # in the sense that each "fiber" contributes). No wait...
    #
    # Let me think more carefully. The tensor M_{n,m,p} is defined by:
    # M_{n,m,p} = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (in appropriate spaces)
    # No, that's not right either.
    #
    # The matrix multiplication tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^{(n)} ⊗ e_j^{(m)} ⊗ e_k^{(p)}
    # where the multiplication is: (A ⊗ B) contracted with T gives AB.
    #
    # Actually, the standard definition: T_{i,j,k} = 1 for all i,j,k is NOT correct.
    # The matrix multiplication tensor has T_{i,j,k} = 1 only when the indices
    # correspond to the multiplication pattern. But in the standard tensor decomposition
    # for matrix multiplication, the tensor IS the all-ones tensor in the sense that
    # every combination (i,j,k) appears exactly once.
    #
    # Let me verify with the trivial decomposition:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for r = i*m*p + j*p + k
    # Then sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (only one r contributes)
    # This equals 1 for all i,j,k.
    #
    # So the tensor we're decomposing is the all-ones tensor of size n x m x p.
    # Its rank is the tensor rank of the all-ones tensor.
    #
    # The tensor rank of the all-ones tensor 1_{n,m,p} is known to be:
    # - For n=m=p=2: rank 7 (Strassen)
    # - For general n,m,p: the rank is at most n*m*p (trivial)
    #
    # For the all-ones tensor, there are known results. The rank of the all-ones
    # tensor 1_{n,m,p} is related to the tensor rank. For small dimensions,
    # the rank is often close to n*m*p but can be lower.
    #
    # Actually, I need to be more careful. The all-ones tensor 1_{n,m,p} has
    # T[i,j,k] = 1 for all i,j,k. Its tensor rank is the minimum r such that
    # T[i,j,k] = sum_{r=0}^{r-1} U[i,r] V[j,r] W[k,r].
    #
    # This is a well-studied object. For 1_{2,2,2}, rank = 7 (Strassen).
    # For 1_{n,m,p} with n,m,p >= 2, the rank is generally between
    # max(n,m,p) and n*m*p.
    #
    # However, I'm not sure the problem is about the all-ones tensor. Let me re-read.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix
    # by a 4 by 7 matrix."
    #
    # The matrix multiplication tensor for multiplying an n x m matrix by an m x p matrix
    # is indeed the tensor T where T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This is because (AB)_{i,k} = sum_j A_{i,j} B_{j,k}, and the tensor decomposition
    # recovers this sum.
    #
    # So we need the tensor rank of the all-ones tensor 1_{3,4,7}.
    #
    # Known bounds for rank of 1_{n,m,p}:
    # - Lower bound: max(n, m, p) (since each slice must have rank >= 1, etc.)
    # - Upper bound: n*m*p (trivial)
    # - For 1_{2,2,2}: rank 7
    # - For 1_{2,2,p}: rank is known for small p
    #
    # For 1_{3,4,7}, I don't know the exact rank. Let me try to find a decomposition
    # with rank less than 84.
    #
    # One approach: use the fact that the all-ones tensor can be written as
    # 1_{n,m,p} = sum_{r} u_r ⊗ v_r ⊗ w_r
    # where u_r, v_r, w_r are vectors.
    #
    # A simple lower-rank decomposition: if we can find vectors u_1,...,u_r in R^n,
    # v_1,...,v_r in R^m, w_1,...,w_r in R^p such that
    # sum_r u_r[i] v_r[j] w_r[k] = 1 for all i,j,k.
    #
    # This is equivalent to saying that the matrix (sum_r u_r[i] v_r[j] w_r[k])_{i,k}
    # as a function of j is... hmm.
    #
    # Actually, a cleaner way: define M_j = sum_r u_r v_r^T ⊗ w_r (outer product structure).
    # We need M_j[i,k] = 1 for all i,j,k.
    #
    # Let me try a specific construction. If we set:
    # u_r = e_{r mod n} (standard basis vector)
    # v_r = e_{(r // n) mod m}
    # w_r = e_{(r // (n*m)) mod p}
    # Then we get the trivial decomposition with rank n*m*p.
    #
    # To get a lower rank, we need to "share" terms. For example, if two terms
    # have the same u and v but different w, we can combine them.
    #
    # Let me try: use rank r = n + m + p - 2 or similar.
    # Actually, for the all-ones tensor, a known decomposition uses rank
    # n + m + p - 2 when one of the dimensions is 2. For general dimensions,
    # the rank is higher.
    #
    # Let me try a randomized search for a lower-rank decomposition.
    # I'll use a simple iterative method: start with a random rank-r decomposition
    # and try to minimize the error.

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        # Compute the reconstructed tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 for all i,j,k
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Try to find a lower-rank decomposition
    # Start with rank = trivial_rank - 1 and go down
    best_rank = trivial_rank
    best_U, best_V, best_W = U, V, W

    # For the all-ones tensor, a known upper bound on rank is:
    # rank(1_{n,m,p}) <= n*m*p - min(n,m,p) + 1 (this is a guess, not sure it's valid)
    # Let me just try a few ranks.

    # Actually, let me try a specific construction.
    # The all-ones tensor 1_{n,m,p} can be decomposed as:
    # 1_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} (e_i ⊗ e_j ⊗ 1_p)
    # where 1_p is the all-ones vector in R^p.
    # This has rank n*m (each term is a rank-1 tensor with w = 1_p).
    # But wait, this gives T[i,j,k] = 1 for all k, which is correct!
    # Let me verify:
    # U[i*m+j, r] where r corresponds to (i,j): U[i*m+j, i*m+j] = 1
    # V[j*p+k, r] = 1 for all k (since v = 1_m... no, v is in R^m)
    #
    # Hmm, I'm confusing myself. Let me be more careful.
    #
    # Tensor decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r in R^n, v_r in R^m, w_r in R^p.
    # T[i,j,k] = sum_r u_r[i] v_r[j] w_r[k].
    #
    # For the all-ones tensor, T[i,j,k] = 1 for all i,j,k.
    #
    # Construction 1: Use n*m terms.
    # For each (i,j), set u_{i,j} = e_i, v_{i,j} = e_j, w_{i,j} = 1_p (all-ones vector).
    # Then T[i',j',k'] = sum_{i,j} e_i[i'] e_j[j'] (1_p)[k'] = sum_{i,j} delta_{i,i'} delta_{j,j'} * 1 = 1.
    # Yes! This works. Rank = n*m = 3*4 = 12.
    #
    # Wait, that seems too good to be true. Let me double-check.
    # T[i',j',k'] = sum_{r} u_r[i'] v_r[j'] w_r[k']
    # With r indexing (i,j) pairs:
    # T[i',j',k'] = sum_{i=0}^{n-1} sum_{j=0}^{m-1} delta_{i,i'} delta_{j,j'} * 1 = 1.
    # Yes, this is correct!
    #
    # So the rank of the all-ones tensor 1_{n,m,p} is at most n*m.
    # By symmetry, it's also at most m*p and n*p.
    # So rank(1_{n,m,p}) <= min(n*m, m*p, n*p).
    #
    # For n=3, m=4, p=7: min(12, 28, 21) = 12.
    #
    # Can we do even better? The rank is at least max(n, m, p) = 7.
    #
    # Let me try to find a decomposition with rank < 12.
    #
    # Actually, let me first verify that the rank-12 decomposition works with the
    # specific indexing convention in the problem.
    #
    # The problem uses:
    # U: (n*m, rank) with U[i*m+j, r]
    # V: (m*p, rank) with V[j*p+k, r]
    # W: (p*n, rank) with W[k*n+i, r]
    #
    # And the decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    #
    # In my construction:
    # For each (i,j) pair, r = i*m + j (0-indexed, so r ranges from 0 to n*m-1 = 11).
    # U[i*m+j, r] = 1 if r == i*m+j, else 0. So U is the identity matrix (n*m x n*m).
    # V[j*p+k, r] = 1 for all j,k,r? No, that's not right.
    #
    # Wait, in my construction, v_{i,j} = e_j, so v_{i,j}[j'] = delta_{j,j'}.
    # But V is indexed by (j,k) pairs, i.e., V[j*p+k, r].
    # The vector v_r is in R^m, so v_r[j] for j in [0,m).
    # But V[j*p+k, r] suggests that V is indexed by both j and k.
    #
    # I think I'm misunderstanding the indexing. Let me re-read the problem.
    #
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"
    #
    # So U is a matrix with rows indexed by (i,j) pairs and columns by r.
    # V is a matrix with rows indexed by (j,k) pairs and columns by r.
    # W is a matrix with rows indexed by (k,i) pairs and columns by r.
    #
    # The decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    #
    # This is NOT the standard tensor decomposition T[i,j,k] = sum_r u_r[i] v_r[j] w_r[k].
    # Instead, it's a "paratensor" or "3-way array" decomposition where:
    # - U has entries indexed by (i,j) and r
    # - V has entries indexed by (j,k) and r
    # - W has entries indexed by (k,i) and r
    #
    # This is equivalent to a tensor decomposition of a 6-dimensional tensor, but
    # it's structured such that the indices are shared between factors.
    #
    # Actually, this is the standard way to represent a bilinear algorithm for matrix
    # multiplication. The tensor being decomposed is the matrix multiplication tensor,
    # which has entries T[i,j,k] = 1 for all i,j,k (as I established earlier).
    #
    # But the decomposition form is different from the standard outer product form.
    # Let me rewrite:
    #
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Let me define:
    # A_{i,j} = U[i*m+j, :] (a vector in R^rank)
    # B_{j,k} = V[j*p+k, :] (a vector in R^rank)
    # C_{k,i} = W[k*n+i, :] (a vector in R^rank)
    #
    # Then the condition is:
    # sum_r A_{i,j}[r] * B_{j,k}[r] * C_{k,i}[r] = 1 for all i,j,k.
    #
    # This is a trilinear form in A, B, C.
    #
    # For the trivial decomposition:
    # A_{i,j} = e_{i*m*p + j*p + k} (one-hot vector of length n*m*p)
    # B_{j,k} = e_{i*m*p + j*p + k}
    # C_{k,i} = e_{i*m*p + j*p + k}
    # Then the sum is 1 (only one term is nonzero).
    #
    # Now, can we find a lower-rank decomposition?
    #
    # Let me try the construction I had before, adapted to this indexing.
    #
    # Construction with rank = n*m = 12:
    # For r = 0, ..., n*m-1, let r = i_r * m + j_r where i_r = r // m, j_r = r % m.
    # Set:
    # U[i*m+j, r] = 1 if (i,j) = (i_r, j_r), else 0. So U is the identity (n*m x n*m).
    # V[j*p+k, r] = 1 for all j,k. (So V is all ones.)
    # W[k*n+i, r] = 1 if i = i_r, else 0. (So W[k*n+i, r] = delta_{i, i_r}.)
    #
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_r delta_{(i,j),(i_r,j_r)} * 1 * delta_{i, i_r}
    # = delta_{(i,j),(i,j)} * 1 * delta_{i,i} = 1.
    #
    # Wait, that's not right. Let me recompute.
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_r delta_{(i,j),(i_r,j_r)} * 1 * delta_{i, i_r}
    #
    # For a given (i,j,k), the sum is over r.
    # U[i*m+j, r] is nonzero only when r = i*m + j, i.e., i_r = i, j_r = j.
    # So the sum reduces to:
    # U[i*m+j, i*m+j] * V[j*p+k, i*m+j] * W[k*n+i, i*m+j]
    # = 1 * 1 * delta_{i, i} = 1.
    #
    # Yes! This works. So rank = n*m = 12 is achievable.
    #
    # By symmetry, we can also achieve rank = m*p = 28 and rank = n*p = 21.
    # The minimum is min(n*m, m*p, n*p) = min(12, 28, 21) = 12.
    #
    # Can we do better than 12? Let me think about lower bounds.
    #
    # The rank must be at least max(n, m, p) = 7.
    #
    # Let me try to find a decomposition with rank < 12.
    #
    # Actually, let me first implement the rank-12 decomposition and verify it.

    # Rank-12 decomposition
    def build_rank_nm():
        r = n * m
        U = np.zeros((n * m, r), dtype=np.float64)
        V = np.zeros((m * p, r), dtype=np.float64)
        W = np.zeros((p * n, r), dtype=np.float64)
        for rr in range(r):
            i_r = rr // m
            j_r = rr % m
            U[i_r * m + j_r, rr] = 1.0
            # V[j*p+k, rr] = 1 for all j,k
            for j in range(m):
                for k in range(p):
                    V[j * p + k, rr] = 1.0
            # W[k*n+i, rr] = 1 if i == i_r
            for k in range(p):
                for i in range(n):
                    if i == i_r:
                        W[k * n + i, rr] = 1.0
        return U, V, W

    U12, V12, W12 = build_rank_nm()
    if check_decomposition(U12, V12, W12):
        best_rank = n * m
        best_U, best_V, best_W = U12, V12, W12
        print(f"Found rank-{best_rank} decomposition")

    # Try rank = n*p = 21
    def build_rank_np():
        r = n * p
        U = np.zeros((n * m, r), dtype=np.float64)
        V = np.zeros((m * p, r), dtype=np.float64)
        W = np.zeros((p * n, r), dtype=np.float64)
        for rr in range(r):
            i_r = rr // p
            k_r = rr % p
            # U[i*m+j, rr] = 1 if i == i_r
            for i in range(n):
                for j in range(m):
                    if i == i_r:
                        U[i * m + j, rr] = 1.0
            # V[j*p+k, rr] = 1 if k == k_r
            for j in range(m):
                for k in range(p):
                    if k == k_r:
                        V[j * p + k, rr] = 1.0
            # W[k*n+i, rr] = 1 if (i,k) == (i_r, k_r)
            W[k_r * n + i_r, rr] = 1.0
        return U, V, W

    U21, V21, W21 = build_rank_np()
    if check_decomposition(U21, V21, W21):
        if n * p < best_rank:
            best_rank = n * p
            best_U, best_V, best_W = U21, V21, W21
            print(f"Found rank-{best_rank} decomposition")

    # Try rank = m*p = 28 (won't be better than 12, but let's check)
    # Skip since 28 > 12

    # Now try to find a decomposition with rank < 12
    # Use a randomized search
    import random
    random.seed(42)

    def random_decomposition(r):
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        return U, V, W

    def error(U, V, W):
        """Compute the maximum error of the decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(val - 1.0))
        return max_err

    # Try ranks from 11 down to 7
    for r in range(best_rank - 1, max(6, best_rank - 5), -1):
        if time.time() > deadline:
            break
        # Try a few random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline:
                break
            U_r, V_r, W_r = random_decomposition(r)
            # Simple gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                # Compute gradient
                dU = np.zeros_like(U_r)
                dV = np.zeros_like(V_r)
                dW = np.zeros_like(W_r)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for rr in range(r):
                                val += U_r[i * m + j, rr] * V_r[j * p + k, rr] * W_r[k * n + i, rr]
                            err = val - 1.0
                            for rr in range(r):
                                dU[i * m + j, rr] += err * V_r[j * p + k, rr] * W_r[k * n + i, rr]
                                dV[j * p + k, rr] += err * U_r[i * m + j, rr] * W_r[k * n + i, rr]
                                dW[k * n + i, rr] += err * U_r[i * m + j, rr] * V_r[j * p + k, rr]
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
            # Check if we found a solution
            if error(U_r, V_r, W_r) < 1e-6:
                # Polish with smaller learning rate
                lr2 = 0.001
                for step in range(500):
                    if time.time() > deadline:
                        break
                    dU = np.zeros_like(U_r)
                    dV = np.zeros_like(V_r)
                    dW = np.zeros_like(W_r)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for rr in range(r):
                                    val += U_r[i * m + j, rr] * V_r[j * p + k, rr] * W_r[k * n + i, rr]
                                err = val - 1.0
                                for rr in range(r):
                                    dU[i * m + j, rr] += err * V_r[j * p + k, rr] * W_r[k * n + i, rr]
                                    dV[j * p + k, rr] += err * U_r[i * m + j, rr] * W_r[k * n + i, rr]
                                    dW[k * n + i, rr] += err * U_r[i * m + j, rr] * V_r[j * p + k, rr]
                    U_r -= lr2 * dU
                    V_r -= lr2 * dV
                    W_r -= lr2 * dW
                if error(U_r, V_r, W_r) < 1e-9:
                    best_rank = r
                    best_U, best_V, best_W = U_r, V_r, W_r
                    print(f"Found rank-{r} decomposition")
                    found = True
                    break
        if found:
            break

    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
