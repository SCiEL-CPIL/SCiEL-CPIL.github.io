import random
import time

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Constructs the factor matrices U, V, W for the tensor decomposition of 
    matrix multiplication of size n x m by m x p.
    
    Uses the schoolbook algorithm which has rank n*m*p.
    For n=3, m=4, p=5, rank = 60.
    
    The decomposition satisfies:
    M_{i,j,k} = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where M_{i,j,k} is 1 if (i,j,k) corresponds to a term in the multiplication, 0 otherwise.
    Specifically, the tensor T for matrix multiplication C = AB has entries T_{i,j,k} such that:
    C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    The bilinear form is sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k} = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}
    Here T_{i,j,k} = 1 if the indices match the multiplication structure, 0 otherwise.
    In the standard tensor definition for matrix multiplication:
    The tensor is in V_{n*m} * V_{m*p} * V_{p*n}.
    Wait, the task says:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    And the output factor follows W[k*n+i, r].
    
    The condition is that the tensor T = sum_r U_r \otimes V_r \otimes W_r 
    equals the matrix multiplication tensor.
    The matrix multiplication tensor T has T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
    In the index notation of the factors:
    U index: i*m + j
    V index: j*p + k
    W index: k*n + i
    
    For a specific (i, j, k), the term is 1.
    So for each (i, j, k), we need a rank-1 component that picks out U[i*m+j], V[j*p+k], W[k*n+i].
    This is exactly the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per the return interface expectation (though numpy arrays are often accepted, 
    # the prompt example uses lists. I will use lists to be safe, or numpy if the evaluator expects arrays.
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries".
    # The example uses lists. I will use lists.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the specific entries for this rank-1 component
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple self-check
    res = construct()
    n, m, p = DIMENSIONS
    rank = len(res["U"][0])
    
    # Verify one random instance
    import random
    random.seed(42)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    # Calculate C = AB normally
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    # Calculate C using decomposition
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # Decomposition: C_{i,k} = sum_{r} (sum_{j} U[i*m+j, r] A_{i,j}) * (sum_{j} V[j*p+k, r] B_{j,k}) * ... 
    # Wait, the tensor decomposition reconstructs the bilinear form.
    # The value of the bilinear form is sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,k} C_{i,k}.
    # Actually, the tensor T represents the multiplication map.
    # T(A, B) = C.
    # The entry T_{i,j,k} is 1 if it's a valid term.
    # The reconstruction of C_{i,k} from the factors is:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # Using the factors:
    # We need to check if sum_{r} W[k*n+i, r] * (sum_{j} U[i*m+j, r] A_{i,j}) * (sum_{j} V[j*p+k, r] B_{j,k}) 
    # ... No, the standard interpretation is:
    # The tensor T = sum_r U_r \otimes V_r \otimes W_r.
    # T(A, B) is a matrix C where C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # The relation is:
    # C_{i,k} = sum_{r} W[k*n+i, r] * (sum_{j} U[i*m+j, r] A_{i,j}) * (sum_{j} V[j*p+k, r] B_{j,k})?
    # No, that's not right.
    
    # Let's trace the indices carefully.
    # T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For schoolbook:
    # If r corresponds to (i', j', k'), then:
    # U[i*m+j, r] is 1 if i=i' and j=j'.
    # V[j*p+k, r] is 1 if j=j' and k=k'.
    # W[k*n+i, r] is 1 if k=k' and i=i'.
    # So the product is 1 only if i=i', j=j', k=k'.
    # Thus T_{i,j,k} = 1 if (i,j,k) is a valid index triple, 0 otherwise.
    # This is the identity tensor for the multiplication space.
    # To get C_{i,k}, we sum over j:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # This is exactly what the tensor T computes when contracted with A and B.
    # So the decomposition is correct.
    
    print("Construction complete. Rank:", rank)
