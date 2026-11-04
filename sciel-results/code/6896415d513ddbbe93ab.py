import numpy as np
from itertools import product
import time

def construct():
    n, m, p = 2, 5, 5
    
    # The tensor T[i,j,k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # We need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c to equal 1
    
    # Strategy:
    # 1. Start with schoolbook rank 50.
    # 2. Try to find a lower rank decomposition using a randomized algebraic search.
    #    We formulate the problem as finding vectors u_c, v_c, w_c such that
    #    sum_c u_c ⊗ v_c ⊗ w_c = T.
    #    This is a system of polynomial equations.
    #    We can use a "sparse" ansatz: assume most entries are 0 or ±1.
    #    Or, we can try to fix some entries and solve the rest.
    
    # Let's try a more direct approach for small ranks.
    # For rank r, we have 45r variables and 50 equations.
    # If we assume a specific sparsity pattern, we might be able to solve it.
    
    # A known technique is to use the "border rank" or "rank" bounds.
    # For 2x5x5, the rank is not trivially known to be < 50.
    # However, we can try to construct a solution with rank 48, 46, etc.
    
    # Let's implement a simple search for a low-rank solution with integer/half-integer entries.
    # We'll try ranks from 50 down to 30.
    # For each rank, we'll try random initializations and use a local search or direct solve.
    
    # Actually, a simpler heuristic:
    # The tensor is all-ones.
    # Rank 1: impossible (needs rank >= 2 for non-zero off-diagonals? No, rank 1 implies T = u⊗v⊗w.
    # T[i,j,k]=1. u[i,j]v[j,k]w[k,i]=1.
    # u[i,j] = 1/v[j,k]w[k,i].
    # u[0,0]v[0,0]w[0,0]=1
    # u[0,0]v[0,1]w[1,0]=1 => v[0,1]w[1,0] = v[0,0]w[0,0]
    # This is possible. Let u=1, v=1, w=1. Then T=1.
    # Wait, if u[i,j]=1, v[j,k]=1, w[k,i]=1, then sum_c ... = 1*1*1 = 1.
    # So rank 1 IS a solution for the all-ones tensor?
    # Let's check the indices.
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # If c=0 only (rank 1):
    # U[i,j] * V[j,k] * W[k,i] = 1 for all i,j,k.
    # Let U[i,j]=1, V[j,k]=1, W[k,i]=1.
    # Then product is 1.
    # So the all-ones tensor has rank 1?
    # No, the tensor for matrix multiplication is NOT the all-ones tensor in the usual basis.
    # The tensor T_{ijk} is 1 if the product A_{ij}B_{jk} contributes to C_{ik}.
    # In the standard basis, T_{ijk} = 1 for all i,j,k?
    # Let's re-read the problem statement.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 5 matrix by a 5 by 5 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # The tensor is the structure tensor of matrix multiplication.
    # T(A,B) = AB.
    # In components: C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T is such that C_{ik} = sum_{j,c} U_{ij,c} V_{jk,c} W_{ki,c} ? No.
    # The decomposition is: C_{ik} = sum_c (sum_j U_{ij,c} A_{ij}) (sum_k V_{jk,c} B_{jk}) ? No.
    # Standard tensor decomposition:
    # C_{ik} = sum_c M_c
    # M_c = (sum_j U_{ij,c} A_{ij}) * (sum_k V_{jk,c} B_{jk}) ? No, that's not right.
    # M_c = (sum_j A_{ij} U_{ij,c}) * (sum_k B_{jk} V_{jk,c}) ?
    # The term is: (u_c^T A) * (v_c^T B) * w_c^T ? No.
    # The rank-1 tensor is u ⊗ v ⊗ w.
    # The contribution to C is: sum_{i,j} u_{ij} A_{ij} * sum_{j,k} v_{jk} B_{jk} * sum_{k,i} w_{ki} C_{ki}?
    # No.
    # The tensor T is defined such that T_{ijk} = 1 if i,j,k are valid.
    # Wait, the structure tensor of matrix multiplication MM(n,m,p) has entries T_{ijk} = 1 for all i,j,k?
    # No. The structure tensor is the trilinear form:
    # <T, A⊗B⊗C> = trace(C^T (AB)) = sum_{i,k} C_{ki} (AB)_{ik} = sum_{i,k} C_{ki} sum_j A_{ij} B_{jk}.
    # = sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # So the coefficients are T_{ijk} = 1 for all i,j,k.
    # So YES, the tensor IS the all-ones tensor in the standard basis!
    # If T_{ijk} = 1 for all i,j,k, then the rank is 1.
    # U_{ij} = 1, V_{jk} = 1, W_{ki} = 1.
    # Let's check the score.
    # If I return rank 1, the score would be 1/50 = 0.02? Or is the score the rank?
    # "score is the rank".
    # "The evaluator measures the raw value (lower is better). The search score is seed/value".
    # If seed is 50 (schoolbook), and value is 1, score = 50/1 = 50?
    # "1.0 equals the seed program".
    # So seed program has rank 50. Score 1.0 = 50/50.
    # If I find rank 1, score = 50/1 = 50.
    # This is a huge improvement.
    
    # Let's verify the rank 1 solution.
    # U is (n*m, rank) = (10, 1).
    # V is (m*p, rank) = (25, 1).
    # W is (p*n, rank) = (10, 1).
    # U[i*m+j, 0] = 1.
    # V[j*p+k, 0] = 1.
    # W[k*n+i, 0] = 1.
    
    # Check: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1*1*1 = 1.
    # This equals T_{ijk} = 1.
    
    # So the minimal rank is 1.
    
    U = [[1] for _ in range(n*m)]
    V = [[1] for _ in range(m*p)]
    W = [[1] for _ in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
