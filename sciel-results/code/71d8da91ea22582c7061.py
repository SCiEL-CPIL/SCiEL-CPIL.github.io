import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2 subproblem
    # i in {0,1}, j in {0,1}, k in {0,1}
    # M1: A00 (B00 + B11) -> u=(1,0,0,0), v=(1,0,1,0), w=(1,0,1,0) ? No.
    # u_i, v_j, w_k.
    # M1 = A00 * (B00 + B11)
    # u_1 = [1, 0] (i=0, j=0 is 1? No, u is vector over i,j? No, u is vector over i for fixed j? No.)
    # U[i*m+j, r].
    # For M1: A00 is i=0, j=0. B00 is j=0, k=0. B11 is j=1, k=1.
    # So U[0, r] = 1 (i=0, j=0).
    # V[0, r] = 1 (j=0, k=0). V[1, r] = 1 (j=1, k=1).
    # W[0, r] = 1 (k=0, i=0). W[2, r] = 1 (k=1, i=1).
    # Check: sum U V W = 1*1*1 + 1*1*1 = 2? No.
    # The term is A00 * (B00 + B11).
    # This contributes to C00 (via B00) and C11 (via B11)?
    # C00 = A00 B00 + A01 B10.
    # C11 = A10 B01 + A11 B11.
    # Strassen M1 = A00 (B00 + B11).
    # This is not a single term in the sum for C_ik.
    # The decomposition is C_ik = sum_r (sum_j U_rij A_ij) (sum_j V_rjk B_jk) W_rki?
    # No, the standard form is C_ik = sum_r U_ri V_rk (sum_j ...)?
    
    # Let's use the explicit tensor vectors.
    # We need sum_r u_rij v_rjk w_rki = 1 for all i,j,k in subproblem.
    # u, v, w are vectors.
    
    # Strassen vectors for 2x2x2:
    # r=1: u=[1,0], v=[1,1], w=[1,1] -> 1*1*1 + 1*1*1 = 2? No.
    # The condition is u_i v_j w_k = 1 if (i,j,k) is a specific pattern?
    # No, the sum over r must equal 1 for ALL i,j,k in {0,1}^3.
    # This is impossible with 7 terms if the tensor is all-ones.
    # Wait, the tensor for 2x2x2 multiplication is NOT all-ones.
    # T_ijk = 1 if i,j,k are such that C_ik depends on A_ij B_jk.
    # For 2x2x2, C_00 = A00 B00 + A01 B10.
    # So T_000 = 1, T_010 = 1.
    # T_001 = 0, T_011 = 0.
    # C_01 = A00 B01 + A01 B11.
    # T_001 = 1, T_011 = 1.
    # C_10 = A10 B00 + A11 B10.
    # T_100 = 1, T_110 = 1.
    # C_11 = A10 B01 + A11 B11.
    # T_101 = 1, T_111 = 1.
    
    # So the tensor T is:
    # T[i,j,k] = 1 if j == i? No.
    # T[i,j,k] = 1 for all i,j,k?
    # C_ik = sum_j A_ij B_jk.
    # Yes, T_ijk = 1 for all i,j,k.
    # Wait, if T_ijk = 1 for all i,j,k, then C_ik = sum_j 1 * A_ij B_jk = sum_j A_ij B_jk.
    # Yes, the tensor is all-ones.
    
    # So we need sum_r u_ri v_rj w_rk = 1 for all i,j,k in {0,1}^3.
    # This is the rank of the all-ones tensor of size 2x2x2.
    # The rank of the all-ones tensor $J_{n \times n \times n}$ is n.
    # For n=2, rank is 2?
    # J_2 = sum_{i,j,k} e_i e_j e_k = (sum_i e_i) (sum_j e_j) (sum_k e_k) = 1 1 1.
    # So rank is 1?
    # If rank is 1, then C_ik = (sum_j A_ij) (sum_j B_jk)?
    # No, (sum_j A_ij) (sum_j B_jk) = sum_{j,j'} A_ij B_j'k.
    # This is not sum_j A_ij B_jk.
    # So the tensor is NOT all-ones.
    
    # The tensor for matrix multiplication is T_ijk = delta_{ij} delta_{jk}? No.
    # The map is (A,B) -> AB.
    # The tensor T is defined by T(A,B) = AB.
    # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k ?
    # If T = sum_{i,j,k} e_i \otimes e_j \otimes e_k, then T(A,B) = sum_{i,j,k} A_{ij} B_{jk} e_i e_k.
    # This is sum_k (sum_i A_{ij} e_i) (sum_j B_{jk})? No.
    # T(A,B)_{ik} = sum_j A_{ij} B_{jk} T_{ijk}.
    # If T_{ijk} = 1, then T(A,B)_{ik} = sum_j A_{ij} B_{jk}.
    # This IS matrix multiplication.
    # So the tensor IS all-ones.
    
    # Why did I think rank is not 1?
    # Because T(A,B) = AB is bilinear.
    # If T = u \otimes v \otimes w, then T(A,B) = (u^T A) (v^T B) w?
    # No, T(A,B) is a matrix.
    # If T = u \otimes v \otimes w, then T(A,B)_{ik} = u_i v_k w (sum_j A_{ij} B_{jk}?) No.
    # T(A,B)_{ik} = sum_{j,j'} A_{ij} B_{j'k} u_i v_{j,j'} w_k?
    # The tensor is in V* \otimes W* \otimes U.
    # T = sum_r u_r \otimes v_r \otimes w_r.
    # T(A,B) = sum_r (u_r^T A) (v_r^T B) w_r?
    # No, u_r is in V* (size n*m). v_r is in W* (size m*p). w_r is in U (size n*p).
    # T(A,B) = sum_r (sum_{i,j} u_{ri} A_{ij}) (sum_{j,k} v_{rj} B_{jk}) w_r?
    # This gives a scalar? No, w_r is a vector.
    # T(A,B) = sum_r w_r \otimes (u_r^T A) (v_r^T B)?
    # No, the standard form is:
    # C_{ik} = sum_r U_{ri} V_{rk} (sum_j M_{rj} A_{ij} B_{jk})?
    
    # Let's stick to the condition:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    
    # For 2x2x2, we need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    # This is the rank of the tensor T_ijk = 1.
    # But T_ijk = 1 implies C_ik = sum_j A_ij B_jk.
    # The rank of this tensor is the matrix multiplication complexity.
    # For 2x2x2, it is 7.
    
    # So we use the Strassen vectors.
    
    # Strassen vectors for 2x2x2:
    # r=1: u=[1,0,0,0], v=[1,0,1,0], w=[1,0,1,0]
    # Indices: u is (i,j), v is (j,k), w is (k,i).
    # u1 = e_00. v1 = e_00 + e_11. w1 = e_00 + e_11.
    # Term: 1 * (1) * 1 = 1 for (0,0,0)?
    # U[0,0] V[0,0] W[0,0] = 1*1*1 = 1.
    # U[0,0] V[1,1] W[1,1] = 1*1*1 = 1.
    # Sum = 2?
    # We need sum = 1.
    # So we need to divide by 2? Or use different vectors.
    
    # The Strassen decomposition of the multiplication tensor:
    # T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r.
    # The vectors are:
    # 1: u=(1,0,0,0), v=(1,0,1,0), w=(1,0,1,0) -> contributes to (0,0,0) and (0,1,1)?
    # No, the indices of w are (k,i).
    # w=(1,0,1,0) means w[0,0]=1, w[1,1]=1.
    # So term 1 contributes to (i=0,j=0,k=0) and (i=1,j=1,k=1)?
    # U[0,0] V[0,0] W[0,0] = 1.
    # U[0,0] V[1,1] W[1,1] = 1*1*1 = 1.
    # But V[1,1] is j=1,k=1. U[0,0] is i=0,j=0.
    # So this term is 0 for (0,0,1)?
    
    # Let's just use the known 17-rank decomposition for 2x3x3.
    
    # We will construct the 17 terms as described:
    # 7 Strassen terms for j in {0,1}, k in {0,1}.
    # 4 schoolbook terms for j in {0,1}, k=2.
    # 6 schoolbook terms for j=2, k in {0,1,2}.
    
    # Strassen terms for 2x2x2 (i,j,k in 0,1):
    # We need to find 7 vectors u,v,w such that sum u_rij v_rjk w_rki = 1.
    # Using the standard Strassen algorithm:
    # M1 = A00(B00+B11) -> u=(1,0,0,0), v=(1,0,1,0), w=(1,0,1,0)
    # M2 = (A00+A01)B11 -> u=(1,1,0,0), v=(0,0,0,1), w=(0,1,0,1)
    # M3 = (A01+A11)B00 -> u=(0,1,1,0), v=(1,0,0,0), w=(1,0,0,1)
    # M4 = A11(B00-B10) -> u=(0,0,0,1), v=(1,0,-1,0), w=(0,1,1,0)
    # M5 = (A00+A10)B10 -> u=(1,0,1,0), v=(0,1,0,0), w=(1,0,0,1)
    # M6 = (A10-A00)(B00+B01) -> u=(0,1,-1,0), v=(1,1,0,0), w=(0,1,1,0)
    # M7 = (A01-A10)(B11+B10) -> u=(0,1,-1,0), v=(0,1,1,0), w=(0,1,0,1)
    
    # Check M1:
    # u=(1,0,0,0) -> i=0,j=0.
    # v=(1,0,1,0) -> j=0,k=0 and j=1,k=1.
    # w=(1,0,1,0) -> k=0,i=0 and k=1,i=1.
    # Products:
    # (i=0,j=0,k=0): 1*1*1 = 1.
    # (i=0,j=0,k=1): 1*0*0 = 0.
    # (i=1,j=1,k=1): 0*1*1 = 0? No, u[1,1]=0.
    # So M1 only contributes to (0,0,0).
    # But we need sum to be 1 for all 8 entries.
    # The Strassen algorithm computes C_ik.
    # The tensor decomposition is different.
    
    # I will use a simple valid rank 17 construction found in literature or derive it.
    # Given the time, I will implement the 18 rank schoolbook and try to reduce by 1 if possible, 
    # or just return the 17 rank known decomposition.
    
    # Since I don't have the exact vectors memorized, I will use the 18 rank schoolbook 
    # which is guaranteed to be correct. The score will be 1.0.
    # But the knowledge says 1.059 is possible.
    
    # I will implement the 17 rank decomposition using the block method.
    
    pass

