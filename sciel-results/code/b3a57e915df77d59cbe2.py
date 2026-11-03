import numpy as np
from functools import lru_cache

def construct():
    """
    Returns a rank-71 exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    Uses recursive block-splitting with Strassen (rank 7) for 2x2x2 sub-blocks.
    """
    # Global dimensions
    n, m, p = 3, 5, 5
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will generate a list of terms.
    # Each term is a tuple (U_dict, V_dict, W_dict)
    # U_dict: {(i, j): val}
    # V_dict: {(j, k): val}
    # W_dict: {(k, i): val}
    # Note: The global indices for U are i*m + j, V is j*p + k, W is k*n + i.
    # The local indices in the recursion are relative to the sub-problem offsets.
    # However, it is easier to keep track of absolute indices or offsets.
    # Let's use a function that returns terms with LOCAL indices (0..n-1, 0..m-1, 0..p-1)
    # and then apply offsets when combining.
    
    # Strassen terms for 2x2x2 (local indices 0,1)
    def strassen_terms():
        # Returns list of (U_dict, V_dict, W_dict)
        # M1 = (a00+a11)(b00+b11) -> C00: +1, C11: +1
        # M2 = (a10+a11)b00       -> C10: +1, C11: -1
        # M3 = a00(b01-b11)       -> C01: +1, C11: +1
        # M4 = a11(b10-b00)       -> C00: +1, C10: +1
        # M5 = (a00+a01)b11       -> C00: -1, C01: +1
        # M6 = (a10-a00)(b00+b01) -> C11: +1
        # M7 = (a01-a11)(b10+b11) -> C00: +1
        
        terms = []
        
        # M1
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        # M2
        terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
        # M3
        terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
        # M4
        terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
        # M5
        terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
        # M6
        terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
        # M7
        terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
        
        return terms

    def schoolbook_terms(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i,j):1}, {(j,k):1}, {(k,i):1}))
        return terms

    @lru_cache(maxsize=None)
    def decompose(n, m, p):
        """
        Returns a tuple of terms (to be hashable for lru_cache).
        Each term is ((U_dict_items), (V_dict_items), (W_dict_items))
        But dicts are not hashable. We need to convert to tuples of tuples.
        """
        # Base cases
        if n == 2 and m == 2 and p == 2:
            # Convert strassen_terms to hashable format
            hashable_terms = []
            for U, V, W in strassen_terms():
                U_t = tuple(sorted(U.items()))
                V_t = tuple(sorted(V.items()))
                W_t = tuple(sorted(W.items()))
                hashable_terms.append((U_t, V_t, W_t))
            return tuple(hashable_terms)
        
        if n <= 1 or m <= 1 or p <= 1:
            hashable_terms = []
            for U, V, W in schoolbook_terms(n, m, p):
                U_t = tuple(sorted(U.items()))
                V_t = tuple(sorted(V.items()))
                W_t = tuple(sorted(W.items()))
                hashable_terms.append((U_t, V_t, W_t))
            return tuple(hashable_terms)
        
        best_terms = None
        best_len = float('inf')
        
        # Try splitting m
        for m1 in range(1, m):
            m2 = m - m1
            terms1 = decompose(n, m1, p)
            terms2 = decompose(n, m2, p)
            
            # Shift terms2: j -> j + m1
            shifted_terms2 = []
            for U_t, V_t, W_t in terms2:
                # U_t: ((i, j), val), V_t: ((j, k), val), W_t: ((k, i), val)
                new_U = tuple(((i, j + m1), val) for (i, j), val in U_t)
                new_V = tuple(((j + m1, k), val) for (j, k), val in V_t)
                # W is unchanged
                shifted_terms2.append((new_U, new_V, W_t))
            
            total = terms1 + tuple(shifted_terms2)
            if len(total) < best_len:
                best_len = len(total)
                best_terms = total
        
        # Try splitting p
        for p1 in range(1, p):
            p2 = p - p1
            terms1 = decompose(n, m, p1)
            terms2 = decompose(n, m, p2)
            
            # Shift terms2: k -> k + p1
            shifted_terms2 = []
            for U_t, V_t, W_t in terms2:
                new_V = tuple(((j, k + p1), val) for (j, k), val in V_t)
                new_W = tuple(((k + p1, i), val) for (k, i), val in W_t)
                # U is unchanged
                shifted_terms2.append((U_t, new_V, new_W))
            
            total = terms1 + tuple(shifted_terms2)
            if len(total) < best_len:
                best_len = len(total)
                best_terms = total
        
        # Try splitting n
        for n1 in range(1, n):
            n2 = n - n1
            terms1 = decompose(n1, m, p)
            terms2 = decompose(n2, m, p)
            
            # Shift terms2: i -> i + n1
            shifted_terms2 = []
            for U_t, V_t, W_t in terms2:
                new_U = tuple(((i + n1, j), val) for (i, j), val in U_t)
                new_W = tuple(((k, i + n1), val) for (k, i), val in W_t)
                # V is unchanged
                shifted_terms2.append((new_U, V_t, new_W))
            
            total = terms1 + tuple(shifted_terms2)
            if len(total) < best_len:
                best_len = len(total)
                best_terms = total
        
        if best_terms is None:
            # Fallback (should not happen for n,m,p > 1)
            hashable_terms = []
            for U, V, W in schoolbook_terms(n, m, p):
                U_t = tuple(sorted(U.items()))
                V_t = tuple(sorted(V.items()))
                W_t = tuple(sorted(W.items()))
                hashable_terms.append((U_t, V_t, W_t))
            return tuple(hashable_terms)
        
        return best_terms

    # Get the decomposition
    terms = decompose(n, m, p)
    rank = len(terms)
    
    # Initialize matrices
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Fill matrices
    for c, (U_t, V_t, W_t) in enumerate(terms):
        # U_t is tuple of ((i, j), val)
        for (i, j), val in U_t:
            U[i*m + j, c] = val
        # V_t is tuple of ((j, k), val)
        for (j, k), val in V_t:
            V[j*p + k, c] = val
        # W_t is tuple of ((k, i), val)
        for (k, i), val in W_t:
            W[k*n + i, c] = val
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple self-test to verify correctness
    res = construct()
    U, V, W = res["U"], res["V"], res["W"]
    n, m, p = 3, 5, 5
    
    # Check tensor decomposition
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should be 1 for all i, j, k
    
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    error += abs(s - 1)
    
    if error < 1e-9:
        print(f"Correct decomposition with rank {U.shape[1]}")
    else:
        print(f"Error in decomposition: {error}")
        print("U shape:", U.shape)
        print("V shape:", V.shape)
        print("W shape:", W.shape)
