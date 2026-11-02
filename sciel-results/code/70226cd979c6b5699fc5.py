import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_shape = (n, m, p)
    # Create the target tensor T
    # T[i, j, k] = 1 if it corresponds to a product term?
    # The tensor M_{n,m,p} is defined such that C_{ik} = sum_j A_{ij} B_{jk}
    # This is equivalent to the tensor T where T[i, j, k] = delta_{i,j} delta_{j,k}? No.
    # The tensor is T[i, j, k] = 1 if the indices match the "identity" of the multiplication.
    # Actually, the tensor is the "structure constant" tensor.
    # For M_{n,m,p}, T[i, j, k] is 1 if i, j, k are valid indices?
    # No, the tensor is T[i, j, k] = 1 for all i, j, k? No.
    # The tensor is T[i, j, k] = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # It contributes if i, j, k are in range.
    # So T[i, j, k] = 1 for all i in [n], j in [m], k in [p].
    # Wait, if T[i, j, k] = 1, then C_{ik} = sum_{j} T[i, j, k] A_{ij} B_{jk}?
    # This would be C_{ik} = sum_j A_{ij} B_{jk}. Yes.
    # So the target tensor is all ones?
    # Let's check the seed code.
    # Seed sets U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # This constructs the tensor where each (i, j, k) is a separate rank-1 term.
    # This corresponds to T[i, j, k] = 1?
    # If T is all ones, then the rank is not nmp.
    # The rank of the all-ones tensor is 1.
    # So the tensor is NOT all ones.
    
    # The tensor M_{n,m,p} has entries T[i, j, k] = 1 if i, j, k are "aligned" in a specific way?
    # No, the standard matrix multiplication tensor is:
    # T[i, j, k] = 1 if the indices correspond to the term A_{ij} B_{jk} -> C_{ik}.
    # But in the tensor product space, the basis is e_i \otimes e_j \otimes e_k.
    # The term A_{ij} B_{jk} is associated with e_i \otimes e_j \otimes e_k.
    # So T[i, j, k] = 1 for all i, j, k?
    # If T is all ones, rank is 1.
    # But we know rank is nmp for schoolbook.
    # Why?
    # Because the decomposition is C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a bilinear map.
    # The tensor is T = sum_{i, j, k} e_i \otimes e_j \otimes e_k?
    # If so, T is the all-ones tensor.
    # But the rank of the all-ones tensor is 1.
    # Contradiction.
    
    # The resolution: The tensor is NOT the all-ones tensor.
    # The tensor M_{n,m,p} is defined by the structure of the multiplication.
    # The basis vectors for A are E_{ij}, for B are F_{jk}, for C are G_{ik}.
    # The multiplication is G_{ik} = sum_j E_{ij} F_{jk}.
    # This defines a tensor T such that <T, E_{ij} \otimes F_{jk} \otimes G_{ik}> = 1?
    # No, the tensor T is the sum of rank-1 tensors u \otimes v \otimes w.
    # The schoolbook decomposition uses u = e_i, v = e_j, w = e_k.
    # So T = sum_{i, j, k} e_i \otimes e_j \otimes e_k.
    # This IS the all-ones tensor.
    # Why is the rank nmp?
    # Ah, the "rank" of matrix multiplication is NOT the tensor rank of the all-ones tensor.
    # It is the rank of the BILINEAR MAP.
    # The bilinear map is A \otimes B -> C.
    # The tensor representing this map is T.
    # The rank of the map is the minimum number of scalar multiplications.
    # This is equal to the tensor rank of T.
    # If T is all-ones, its rank is 1.
    # So T is NOT all-ones.
    
    # Let's re-read the definition.
    # T[i, j, k] = 1 if the term A_{ij} B_{jk} is used to compute C_{ik}.
    # In standard multiplication, EVERY term A_{ij} B_{jk} is used.
    # So T[i, j, k] = 1 for all i, j, k.
    # This implies T is all-ones.
    # But rank(all-ones) = 1.
    # This is a well-known confusion.
    # The tensor M_{n,m,p} is NOT the all-ones tensor.
    # The all-ones tensor corresponds to C_{ik} = (sum A_{ij}) (sum B_{jk}).
    # Standard multiplication is C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a DIFFERENT tensor.
    # The tensor for standard multiplication is:
    # T[i, j, k] = 1 if the indices are "compatible"?
    # No, the tensor is T = sum_{i, j, k} e_i \otimes e_j \otimes e_k is WRONG.
    # The correct tensor is T = sum_{i, j, k} e_i \otimes e_j \otimes e_k is the tensor for (sum A)(sum B).
    # The tensor for C_{ik} = sum_j A_{ij} B_{jk} is:
    # T[i, j, k] = 1 if i, j, k are such that...?
    # Actually, the tensor is T[i, j, k] = 1 for all i, j, k?
    # No.
    # Let's look at the seed code again.
    # U[i*m+j, c] = 1.
    # V[j*p+k, c] = 1.
    # W[k*n+i, c] = 1.
    # This creates a tensor where the term for (i, j, k) is e_{im+j} \otimes e_{jp+k} \otimes e_{kn+i}.
    # This is a permutation of the all-ones tensor?
    # No, it's a specific set of basis vectors.
    # The tensor is T = sum_{i, j, k} u_{ijk} \otimes v_{ijk} \otimes w_{ijk}.
    # This T is the "identity" tensor in the sense that it maps A \otimes B to C correctly.
    # The rank of this T is nmp.
    
    # So I need to find a lower rank decomposition of THIS specific tensor.
    # The tensor is T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{im+j} \otimes e_{jp+k} \otimes e_{kn+i}.
    # Note the indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # I will construct this target tensor and search for a lower rank decomposition.
    
    target_tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_tensor[i*m+j, j*p+k, k*n+i] = 1.0

    def verify_decomp(U, V, W):
        # Check if U \otimes V \otimes W equals target_tensor
        # Flatten and compare
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        # The indices in einsum correspond to U rows, V rows, W rows.
        # U rows are 0..nm-1, V rows 0..mp-1, W rows 0..pn-1.
        # So recon has shape (nm, mp, pn).
        # This matches target_tensor shape.
        return np.allclose(recon, target_tensor, atol=1e-9)

    def try_find_decomp(rank):
        # Random restarts
        for _ in range(5):
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            def residual(params):
                u, v, w = params[:n*m*rank], params[n*m*rank:n*m*rank+m*p*rank], params[n*m*rank+m*p*rank:]
                u = u.reshape(n*m, rank)
                v = v.reshape(m*p, rank)
                w = w.reshape(p*n, rank)
                recon = np.einsum('ir,jr,kr->ijk', u, v, w)
                return (recon - target_tensor).flatten()
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            res = least_squares(residual, x0, method='lm', max_nfev=1000)
            if res.cost < 1e-10:
                u = res.x[:n*m*rank].reshape(n*m, rank)
                v = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                w = res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                # Check for rationality
                # Convert to fractions
                from fractions import Fraction
                U_frac = [[Fraction(u[i, r]).limit_denominator(1000) for r in range(rank)] for i in range(n*m)]
                V_frac = [[Fraction(v[i, r]).limit_denominator(1000) for r in range(rank)] for i in range(m*p)]
                W_frac = [[Fraction(w[i, r]).limit_denominator(1000) for r in range(rank)] for i in range(p*n)]
                
                # Verify with fractions
                recon_frac = np.zeros((n*m, m*p, p*n), dtype=object)
                for i in range(n*m):
                    for j in range(m*p):
                        for k in range(p*n):
                            s = Fraction(0)
                            for r in range(rank):
                                s += U_frac[i][r] * V_frac[j][r] * W_frac[k][r]
                            recon_frac[i, j, k] = s
                            
                if np.all([recon_frac[i, j, k] == target_tensor[i, j, k] for i in range(n*m) for j in range(m*p) for k in range(p*n)]):
                    return U_frac, V_frac, W_frac
        return None

    start_time = time.time()
    for rank in range(17, 31):
        if time.time() - start_time > 120:
            break
        decomp = try_find_decomp(rank)
        if decomp:
            U, V, W = decomp
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
