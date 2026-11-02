import numpy as np
import time

START_TIME = time.time()
BUDGET_SECONDS = 135.0

def check_time():
    return time.time() - START_TIME < BUDGET_SECONDS

def construct():
    n, m, p = 3, 4, 5
    
    # Helper to create a zeroed decomposition
    def create_decomp(rank):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        return U, V, W

    # Recursive function to get decomposition for n, m, p
    # Returns a list of triples (u_vec, v_vec, w_vec)
    # u_vec is length n*m, v_vec is length m*p, w_vec is length p*n
    
    def get_local_decomp(n_sub, m_sub, p_sub):
        """
        Returns a list of (U_sub, V_sub, W_sub) where:
        U_sub is n_sub x m_sub
        V_sub is m_sub x p_sub
        W_sub is p_sub x n_sub
        """
        # Base case: Schoolbook
        if n_sub == 1 and m_sub == 1 and p_sub == 1:
            U = np.array([[1]])
            V = np.array([[1]])
            W = np.array([[1]])
            return [(U, V, W)]
        
        # If any dimension is 1, we can reduce to a product of the other two
        if n_sub == 1:
            # C = A * B, where A is 1 x m, B is m x p
            # This is just a dot product, rank = m
            terms = []
            for j in range(m_sub):
                U = np.zeros((1, m_sub))
                V = np.zeros((m_sub, p_sub))
                W = np.zeros((p_sub, 1))
                U[0, j] = 1
                V[j, :] = 1
                W[:, 0] = 1
                terms.append((U, V, W))
            return terms
            
        if m_sub == 1:
            # C = A * B, where A is n x 1, B is 1 x p
            # C[i,k] = A[i,0] * B[0,k]
            # Rank = n * p
            terms = []
            for i in range(n_sub):
                for k in range(p_sub):
                    U = np.zeros((n_sub, 1))
                    V = np.zeros((1, p_sub))
                    W = np.zeros((p_sub, n_sub))
                    U[i, 0] = 1
                    V[0, k] = 1
                    W[k, i] = 1
                    terms.append((U, V, W))
            return terms
            
        if p_sub == 1:
            # C = A * B, where A is n x m, B is m x 1
            # C[i,0] = sum_j A[i,j] B[j,0]
            # Rank = n * m
            terms = []
            for i in range(n_sub):
                for j in range(m_sub):
                    U = np.zeros((n_sub, m_sub))
                    V = np.zeros((m_sub, 1))
                    W = np.zeros((1, n_sub))
                    U[i, j] = 1
                    V[j, 0] = 1
                    W[0, i] = 1
                    terms.append((U, V, W))
            return terms

        # Special case: Strassen for 2x2x2
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            # Strassen's 7 terms
            terms = []
            
            # 1
            U1 = np.array([[1, 0], [0, 1]])
            V1 = np.array([[1, 0], [0, 1]])
            W1 = np.array([[1, 0], [0, 1]]) 
            terms.append((U1, V1, W1))
            
            # 2
            U2 = np.array([[0, 0], [1, 1]])
            V2 = np.array([[1, 0], [0, 0]])
            W2 = np.array([[0, 1], [-1, 0]]) 
            terms.append((U2, V2, W2))
            
            # 3
            U3 = np.array([[1, 0], [0, 0]])
            V3 = np.array([[-1, 1], [0, 0]])
            W3 = np.array([[0, 0], [1, 1]]) 
            terms.append((U3, V3, W3))
            
            # 4
            U4 = np.array([[0, 0], [0, 1]])
            V4 = np.array([[0, 0], [-1, 1]])
            W4 = np.array([[1, 1], [0, 0]]) 
            terms.append((U4, V4, W4))
            
            # 5
            U5 = np.array([[1, 1], [0, 0]])
            V5 = np.array([[0, 0], [0, 1]])
            W5 = np.array([[-1, 0], [1, 0]]) 
            terms.append((U5, V5, W5))
            
            # 6
            U6 = np.array([[-1, 0], [1, 0]])
            V6 = np.array([[1, 0], [1, 0]])
            W6 = np.array([[0, 0], [0, 1]]) 
            terms.append((U6, V6, W6))
            
            # 7
            U7 = np.array([[0, 1], [0, -1]])
            V7 = np.array([[0, 1], [0, 1]])
            W7 = np.array([[1, 0], [0, 0]]) 
            terms.append((U7, V7, W7))
            
            return terms

        # Recursive case: Split the largest dimension or a specific one
        # Strategy:
        # If m > 1, split m into m1, m2.
        # M(n, m, p) = M(n, m1, p) + M(n, m2, p)
        
        if m_sub > 1:
            # Choose split size to optimize for known fast cases (like 2x2x2)
            # For m=4, split into 2 and 2
            # For m=3, split into 1 and 2
            if m_sub == 4:
                m1 = 2
            elif m_sub == 3:
                m1 = 1
            else:
                m1 = m_sub // 2
            m2 = m_sub - m1
            
            terms1 = get_local_decomp(n_sub, m1, p_sub)
            terms2 = get_local_decomp(n_sub, m2, p_sub)
            
            all_terms = []
            # Embed terms1
            for U, V, W in terms1:
                U_full = np.zeros((n_sub, m_sub))
                V_full = np.zeros((m_sub, p_sub))
                W_full = W # W is p x n, no change in j
                U_full[:, :m1] = U
                V_full[:m1, :] = V
                all_terms.append((U_full, V_full, W_full))
            
            # Embed terms2
            for U, V, W in terms2:
                U_full = np.zeros((n_sub, m_sub))
                V_full = np.zeros((m_sub, p_sub))
                W_full = W
                U_full[:, m1:] = U
                V_full[m1:, :] = V
                all_terms.append((U_full, V_full, W_full))
                
            return all_terms
        
        # If n > 1, split n
        if n_sub > 1:
            # For n=3, split into 1 and 2
            if n_sub == 3:
                n1 = 1
            else:
                n1 = n_sub // 2
            n2 = n_sub - n1
            
            terms1 = get_local_decomp(n1, m_sub, p_sub)
            terms2 = get_local_decomp(n2, m_sub, p_sub)
            
            all_terms = []
            # Embed terms1
            for U, V, W in terms1:
                U_full = np.zeros((n_sub, m_sub))
                V_full = V # V is m x p, no change in i
                W_full = np.zeros((p_sub, n_sub))
                U_full[:n1, :] = U
                W_full[:, :n1] = W
                all_terms.append((U_full, V_full, W_full))
            
            # Embed terms2
            for U, V, W in terms2:
                U_full = np.zeros((n_sub, m_sub))
                V_full = V
                W_full = np.zeros((p_sub, n_sub))
                U_full[n1:, :] = U
                W_full[:, n1:] = W
                all_terms.append((U_full, V_full, W_full))
                
            return all_terms
        
        # If p > 1, split p
        if p_sub > 1:
            # For p=5, split into 2 and 3 (or 2 and 3)
            # Let's try splitting to get 2 and 3 to use Strassen on the 2 part?
            # Or maybe 3 and 2.
            if p_sub == 5:
                p1 = 2
            else:
                p1 = p_sub // 2
            p2 = p_sub - p1
            
            terms1 = get_local_decomp(n_sub, m_sub, p1)
            terms2 = get_local_decomp(n_sub, m_sub, p2)
            
            all_terms = []
            # Embed terms1
            for U, V, W in terms1:
                U_full = U
                V_full = np.zeros((m_sub, p_sub))
                W_full = np.zeros((p_sub, n_sub))
                V_full[:, :p1] = V
                W_full[:p1, :] = W
                all_terms.append((U_full, V_full, W_full))
            
            # Embed terms2
            for U, V, W in terms2:
                U_full = U
                V_full = np.zeros((m_sub, p_sub))
                W_full = np.zeros((p_sub, n_sub))
                V_full[:, p1:] = V
                W_full[p1:, :] = W
                all_terms.append((U_full, V_full, W_full))
                
            return all_terms
        
        # Fallback (should not reach here for n,m,p >= 1)
        return []

    # We want to try different splitting strategies to find the minimum rank.
    # The recursive function above uses a greedy strategy.
    # Let's implement a more systematic search for the optimal split.
    
    # Known ranks for small matrices:
    # R(1,1,1) = 1
    # R(1,m,p) = m
    # R(n,1,p) = n*p
    # R(n,m,1) = n*m
    # R(2,2,2) = 7
    # R(2,2,3) = ? (Maybe 10? 12?)
    # R(2,3,3) = ?
    
    # Let's use dynamic programming to find the minimum rank.
    # R(n, m, p) = min over splits (R(n1, m, p) + R(n2, m, p), R(n, m1, p) + R(n, m2, p), R(n, m, p1) + R(n, m, p2))
    # Base cases as above.
    
    def compute_min_rank(n, m, p, memo={}):
        if (n, m, p) in memo:
            return memo[(n, m, p)]
        
        # Base cases
        if n == 1 and m == 1 and p == 1:
            memo[(n, m, p)] = 1
            return 1
        if n == 1:
            r = m
            memo[(n, m, p)] = r
            return r
        if m == 1:
            r = n * p
            memo[(n, m, p)] = r
            return r
        if p == 1:
            r = n * m
            memo[(n, m, p)] = r
            return r
        
        # Special cases
        if n == 2 and m == 2 and p == 2:
            memo[(n, m, p)] = 7
            return 7
            
        # Try all splits
        best = float('inf')
        
        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            r1 = compute_min_rank(n1, m, p)
            r2 = compute_min_rank(n2, m, p)
            best = min(best, r1 + r2)
            
        # Split m
        for m1 in range(1, m):
            m2 = m - m1
            r1 = compute_min_rank(n, m1, p)
            r2 = compute_min_rank(n, m2, p)
            best = min(best, r1 + r2)
            
        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            r1 = compute_min_rank(n, m, p1)
            r2 = compute_min_rank(n, m, p2)
            best = min(best, r1 + r2)
            
        memo[(n, m, p)] = best
        return best

    min_rank = compute_min_rank(n, m, p)
    
    # Now we need to construct the decomposition with that rank.
    # We can modify get_local_decomp to use the optimal splits determined by the DP.
    
    def get_optimal_split(n, m, p):
        # Returns (dim, split_size) or None if base case
        if n == 1 and m == 1 and p == 1:
            return None
        if n == 1:
            return ('m', m) # This is not a split, it's a base case handling
        if m == 1:
            return ('n', n)
        if p == 1:
            return ('m', m)
        if n == 2 and m == 2 and p == 2:
            return ('strassen', None)
            
        best_rank = float('inf')
        best_split = None
        
        # Split n
        for n1 in range(1, n):
            n2 = n - n1
            r1 = compute_min_rank(n1, m, p)
            r2 = compute_min_rank(n2, m, p)
            total = r1 + r2
            if total < best_rank:
                best_rank = total
                best_split = ('n', n1)
                
        # Split m
        for m1 in range(1, m):
            m2 = m - m1
            r1 = compute_min_rank(n, m1, p)
            r2 = compute_min_rank(n, m2, p)
            total = r1 + r2
            if total < best_rank:
                best_rank = total
                best_split = ('m', m1)
                
        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            r1 = compute_min_rank(n, m, p1)
            r2 = compute_min_rank(n, m, p2)
            total = r1 + r2
            if total < best_rank:
                best_rank = total
                best_split = ('p', p1)
                
        return best_split

    def get_decomp_optimal(n_sub, m_sub, p_sub):
        # Returns list of (U, V, W)
        split_info = get_optimal_split(n_sub, m_sub, p_sub)
        
        if split_info is None:
            # 1x1x1
            return [(np.array([[1]]), np.array([[1]]), np.array([[1]]))]
            
        if n_sub == 1:
            terms = []
            for j in range(m_sub):
                U = np.zeros((1, m_sub))
                V = np.zeros((m_sub, p_sub))
                W = np.zeros((p_sub, 1))
                U[0, j] = 1
                V[j, :] = 1
                W[:, 0] = 1
                terms.append((U, V, W))
            return terms
            
        if m_sub == 1:
            terms = []
            for i in range(n_sub):
                for k in range(p_sub):
                    U = np.zeros((n_sub, 1))
                    V = np.zeros((1, p_sub))
                    W = np.zeros((p_sub, n_sub))
                    U[i, 0] = 1
                    V[0, k] = 1
                    W[k, i] = 1
                    terms.append((U, V, W))
            return terms
            
        if p_sub == 1:
            terms = []
            for i in range(n_sub):
                for j in range(m_sub):
                    U = np.zeros((n_sub, m_sub))
                    V = np.zeros((m_sub, 1))
                    W = np.zeros((1, n_sub))
                    U[i, j] = 1
                    V[j, 0] = 1
                    W[0, i] = 1
                    terms.append((U, V, W))
            return terms
            
        if split_info[0] == 'strassen':
            return get_local_decomp(2, 2, 2) # Reuse the Strassen code
            
        dim, size1 = split_info
        size2 = (n_sub if dim == 'n' else m_sub if dim == 'm' else p_sub) - size1
        
        if dim == 'n':
            terms1 = get_decomp_optimal(size1, m_sub, p_sub)
            terms2 = get_decomp_optimal(size2, m_sub, p_sub)
            
            all_terms = []
            for U, V, W in terms1:
                U_full = np.zeros((n_sub, m_sub))
                V_full = V
                W_full = np.zeros((p_sub, n_sub))
                U_full[:size1, :] = U
                W_full[:, :size1] = W
                all_terms.append((U_full, V_full, W_full))
            
            for U, V, W in terms2:
                U_full = np.zeros((n_sub, m_sub))
                V_full = V
                W_full = np.zeros((p_sub, n_sub))
                U_full[size1:, :] = U
                W_full[:, size1:] = W
                all_terms.append((U_full, V_full, W_full))
                
            return all_terms
            
        elif dim == 'm':
            terms1 = get_decomp_optimal(n_sub, size1, p_sub)
            terms2 = get_decomp_optimal(n_sub, size2, p_sub)
            
            all_terms = []
            for U, V, W in terms1:
                U_full = np.zeros((n_sub, m_sub))
                V_full = np.zeros((m_sub, p_sub))
                W_full = W
                U_full[:, :size1] = U
                V_full[:size1, :] = V
                all_terms.append((U_full, V_full, W_full))
            
            for U, V, W in terms2:
                U_full = np.zeros((n_sub, m_sub))
                V_full = np.zeros((m_sub, p_sub))
                W_full = W
                U_full[:, size1:] = U
                V_full[size1:, :] = V
                all_terms.append((U_full, V_full, W_full))
                
            return all_terms
            
        elif dim == 'p':
            terms1 = get_decomp_optimal(n_sub, m_sub, size1)
            terms2 = get_decomp_optimal(n_sub, m_sub, size2)
            
            all_terms = []
            for U, V, W in terms1:
                U_full = U
                V_full = np.zeros((m_sub, p_sub))
                W_full = np.zeros((p_sub, n_sub))
                V_full[:, :size1] = V
                W_full[:size1, :] = W
                all_terms.append((U_full, V_full, W_full))
            
            for U, V, W in terms2:
                U_full = U
                V_full = np.zeros((m_sub, p_sub))
                W_full = np.zeros((p_sub, n_sub))
                V_full[:, size1:] = V
                W_full[size1:, :] = W
                all_terms.append((U_full, V_full, W_full))
                
            return all_terms
            
        return []

    local_terms = get_decomp_optimal(n, m, p)
    rank = len(local_terms)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, (U_sub, V_sub, W_sub) in enumerate(local_terms):
        # Flatten U_sub (n x m) to length n*m
        # Row i, Col j -> index i*m + j
        U[:, r] = U_sub.flatten(order='C')
        
        # Flatten V_sub (m x p) to length m*p
        # Row j, Col k -> index j*p + k
        V[:, r] = V_sub.flatten(order='C')
        
        # Flatten W_sub (p x n) to length p*n
        # Row k, Col i -> index k*n + i
        W[:, r] = W_sub.flatten(order='C')
        
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
