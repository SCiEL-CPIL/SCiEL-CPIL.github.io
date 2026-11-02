import numpy as np
import time

START_TIME = time.time()
BUDGET_SECONDS = 140.0

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
    # But it's easier to work with local indices and then embed.
    
    def get_local_decomp(n_sub, m_sub, p_sub):
        """
        Returns a list of (U_sub, V_sub, W_sub) where:
        U_sub is n_sub x m_sub
        V_sub is m_sub x p_sub
        W_sub is p_sub x n_sub
        """
        # Base case: Schoolbook
        if n_sub == 1 or m_sub == 1 or p_sub == 1:
            terms = []
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U = np.zeros((n_sub, m_sub))
                        V = np.zeros((m_sub, p_sub))
                        W = np.zeros((p_sub, n_sub))
                        U[i, j] = 1
                        V[j, k] = 1
                        W[k, i] = 1
                        terms.append((U, V, W))
            return terms
        
        # Special case: Strassen for 2x2x2
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            # Strassen's 7 terms
            # m1 = (A00+A11)(B00+B11)
            # m2 = (A10+A11)(B00)
            # m3 = (A00)(B01-B00)
            # m4 = (A11)(B11-B10)
            # m5 = (A00+A01)(B11)
            # m6 = (A10-A00)(B00+B10)
            # m7 = (A01-A11)(B01+B11)
            
            terms = []
            
            # 1
            U1 = np.array([[1, 0], [0, 1]])
            V1 = np.array([[1, 0], [0, 1]])
            W1 = np.array([[1, 0], [0, 1]]) # C00 += 1, C11 += 1. W[k,i]. W[0,0]=1, W[1,1]=1.
            terms.append((U1, V1, W1))
            
            # 2
            U2 = np.array([[0, 0], [1, 1]])
            V2 = np.array([[1, 0], [0, 0]])
            W2 = np.array([[0, 1], [-1, 0]]) # C10 += 1, C11 -= 1. W[0,1]=1, W[1,1]=-1.
            terms.append((U2, V2, W2))
            
            # 3
            U3 = np.array([[1, 0], [0, 0]])
            V3 = np.array([[-1, 1], [0, 0]])
            W3 = np.array([[0, 0], [1, 1]]) # C01 += 1, C11 += 1. W[1,0]=1, W[1,1]=1.
            terms.append((U3, V3, W3))
            
            # 4
            U4 = np.array([[0, 0], [0, 1]])
            V4 = np.array([[0, 0], [-1, 1]])
            W4 = np.array([[1, 1], [0, 0]]) # C00 += 1, C10 += 1. W[0,0]=1, W[0,1]=1.
            terms.append((U4, V4, W4))
            
            # 5
            U5 = np.array([[1, 1], [0, 0]])
            V5 = np.array([[0, 0], [0, 1]])
            W5 = np.array([[-1, 0], [1, 0]]) # C00 -= 1, C01 += 1. W[0,0]=-1, W[1,0]=1.
            terms.append((U5, V5, W5))
            
            # 6
            U6 = np.array([[-1, 0], [1, 0]])
            V6 = np.array([[1, 0], [1, 0]])
            W6 = np.array([[0, 0], [0, 1]]) # C11 += 1. W[1,1]=1.
            terms.append((U6, V6, W6))
            
            # 7
            U7 = np.array([[0, 1], [0, -1]])
            V7 = np.array([[0, 1], [0, 1]])
            W7 = np.array([[1, 0], [0, 0]]) # C00 += 1. W[0,0]=1.
            terms.append((U7, V7, W7))
            
            return terms
        
        # Recursive case: Split the largest dimension or a specific one
        # Strategy:
        # If m > 1, split m into m1, m2.
        # M(n, m, p) = M(n, m1, p) + M(n, m2, p)
        # Indices:
        # First part uses j in [0, m1)
        # Second part uses j in [m1, m)
        
        if m_sub > 1:
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

    local_terms = get_local_decomp(n, m, p)
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
