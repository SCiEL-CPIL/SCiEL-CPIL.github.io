"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block size 2, number of blocks 3
    block_size = 2
    num_blocks = 3
    rank = num_blocks**3 * 7  # 27 * 7 = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen definitions for 2x2 blocks
    # Indices i, j in {0, 1}
    # U indices: (I*2+i, J*2+j) -> flat index (I*2+i)*6 + (J*2+j)
    # V indices: (J*2+j, K*2+k) -> flat index (J*2+j)*6 + (K*2+k)
    # W indices: (K*2+k, I*2+i) -> flat index (K*2+k)*6 + (I*2+i)
    
    # The 7 Strassen products M1..M7
    # Each product is defined by linear combinations of A and B blocks
    # We represent them as lists of (index, coeff) for U and V
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: (0,0)+1, (1,1)+1
    # V: (0,0)+1, (0,1)+1, (1,0)+1, (1,1)+1 -> Wait, B is 2x2.
    # Let's be precise.
    # A is 2x2, B is 2x2.
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    
    # M1: A00+A11 times B00+B11
    # Actually, it's (A00+A11) * (B00+B11).
    # This is not a single rank-1 term in the tensor sense unless we expand.
    # Wait, Strassen's algorithm works by computing 7 products of sums.
    # The "rank" of the matrix multiplication tensor is the number of scalar multiplications.
    # Each scalar multiplication corresponds to one rank-1 term in the tensor decomposition?
    # No. The tensor $T_{ijk} = \delta_{ik}$ for $n \times m \times p$? No, $T_{ijk}$ is the coefficient of $A_{ij} B_{jk}$ in $C_{ik}$.
    # The decomposition is $T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
    # $C_{ik} = \sum_{j} A_{ij} B_{jk} = \sum_{r} U_{ir} V_{jr} W_{kr}$? No.
    # Standard form: $C_{ik} = \sum_{r} W_{ik}^{(r)} (\sum_{j} U_{ij}^{(r)} A_{ij}) (\sum_{j} V_{jk}^{(r)} B_{jk})$? No.
    # The factorization is $A \cdot B = \sum_{r} W_r (U_r \cdot A) (V_r \cdot B)$? No.
    # It is $C_{ik} = \sum_{r} \alpha_{ik}^{(r)} (\sum_{j} \beta_{ij}^{(r)} A_{ij}) (\sum_{j} \gamma_{jk}^{(r)} B_{jk})$?
    # Usually: $C_{ik} = \sum_{r} W_{ik} [r] \cdot ( \sum_{j} U_{ij}[r] A_{ij} ) \cdot ( \sum_{j} V_{jk}[r] B_{jk} )$? No.
    # The tensor is $T \in \mathbb{R}^{n \times m \times p}$. $T_{i,j,k}$ is 1 if $i=k$ and any $j$? No.
    # $T_{i,j,k}$ is the coefficient of $A_{i,j} B_{j,k}$ in the product.
    # So $T_{i,j,k} = 1$ if $i=k$? No.
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # So $T_{i,j,k} = 1$ for all $i,j,k$? No.
    # The tensor $T$ has entries $T_{i,j,k} = 1$ if the term $A_{i,j} B_{j,k}$ appears in $C_{i,k}$.
    # It appears exactly once. So $T_{i,j,k} = 1$ for all $i,j,k$?
    # Wait, $C_{i,k}$ depends on $A_{i,j}$ and $B_{j,k}$.
    # So the tensor is $T_{i,j,k} = 1$ for all $i,j,k$?
    # Let's check the rank-1 decomposition.
    # $T = \sum_{r} u_r \otimes v_r \otimes w_r$.
    # $T_{i,j,k} = \sum_{r} U_{i,r} V_{j,r} W_{k,r}$? No.
    # The indices are $i$ (row of A), $j$ (col of A / row of B), $k$ (col of B).
    # So $U$ has indices $(i, r)$, $V$ has indices $(j, r)$, $W$ has indices $(k, r)$.
    # Then $T_{i,j,k} = \sum_{r} U_{i,r} V_{j,r} W_{k,r}$.
    # For schoolbook: $T_{i,j,k} = 1$.
    # Rank $nmp$. $U_{i,r}=1, V_{j,r}=1, W_{k,r}=1$ for specific $r$.
    
    # For Strassen:
    # We are decomposing the $2 \times 2 \times 2$ tensor of rank 7.
    # Then we tensor it with the identity on the block indices?
    # Yes.
    
    # Let's define the 7 vectors for the $2 \times 2 \times 2$ case.
    # $i, j, k \in \{0, 1\}$.
    # $U$ shape $(2, 7)$, $V$ shape $(2, 7)$, $W$ shape $(2, 7)$.
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # Wait, the output is $C$.
    # $C_{00} = M_1 + M_4 - M_5 + M_7$
    # $C_{01} = M_3 + M_5$
    # $C_{10} = M_2 + M_4$
    # $C_{11} = M_1 - M_2 + M_3 + M_6$
    
    # Let's list the coefficients for each $M_r$.
    # r=0 (M1):
    # A part: A00+A11. U: [1, 0, 1, 0] (flattened i,j)
    # B part: B00+B11. V: [1, 0, 0, 1] (flattened j,k)
    # C part: C00+C11. W: [1, 0, 0, 1] (flattened k,i)
    
    # r=1 (M2):
    # A part: A10+A11. U: [0, 0, 1, 1]
    # B part: B00. V: [1, 0, 0, 0]
    # C part: C10+C11. W: [0, 0, 1, 1]
    
    # r=2 (M3):
    # A part: A00. U: [1, 0, 0, 0]
    # B part: B10-B11. V: [0, 0, 1, -1]
    # C part: C01+C11. W: [0, 1, 0, 1]
    
    # r=3 (M4):
    # A part: A11. U: [0, 0, 0, 1]
    # B part: B01-B00. V: [0, 1, -1, 0]
    # C part: C00+C10. W: [1, 0, 1, 0]
    
    # r=4 (M5):
    # A part: A00+A01. U: [1, 1, 0, 0]
    # B part: B11. V: [0, 0, 0, 1]
    # C part: -C00+C01. W: [-1, 1, 0, 0]
    
    # r=5 (M6):
    # A part: A10-A00. U: [-1, 0, 1, 0]
    # B part: B00+B01. V: [1, 1, 0, 0]
    # C part: C11. W: [0, 0, 0, 1]
    
    # r=6 (M7):
    # A part: A01-A11. U: [0, 1, 0, -1]
    # B part: B10+B11. V: [0, 0, 1, 1]
    # C part: C00. W: [1, 0, 0, 0]
    
    # Note on W indices: $W$ maps to $C_{k,i}$?
    # The task says $W[k*n+i, r]$.
    # So $W$ index is $(k, i)$.
    # My flattened $W$ above was for $C_{k,i}$?
    # $C$ is $n \times p$? No, $C$ is $n \times p$ where $n=6, p=6$.
    # $C_{i,k}$.
    # $W$ index is $k*n + i$.
    # So $W$ corresponds to $C_{i,k}$ but stored as $W[k,i]$.
    # Let's re-verify the W mapping.
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # Decomposition: $C_{i,k} = \sum_r W_{k,i}^{(r)} ( \sum_j U_{i,j}^{(r)} A_{i,j} ) ( \sum_j V_{j,k}^{(r)} B_{j,k} )$?
    # No, the tensor is $T_{i,j,k}$.
    # $T_{i,j,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
    # Here $U$ is $n \times R$, $V$ is $m \times R$, $W$ is $p \times R$.
    # Wait, the task says:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # This is a different flattening.
    # $U$ has index $i*m+j$.
    # $V$ has index $j*p+k$.
    # $W$ has index $k*n+i$.
    
    # So $U_{i*m+j, r}$, $V_{j*p+k, r}$, $W_{k*n+i, r}$.
    # The tensor entry $T_{i,j,k}$ is decomposed as $\sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}$.
    # This matches the standard tensor definition $T_{i,j,k}$.
    
    # So for the $2 \times 2 \times 2$ block:
    # $i, j, k \in \{0, 1\}$.
    # $U$ index: $i*2+j$.
    # $V$ index: $j*2+k$.
    # $W$ index: $k*2+i$.
    
    # Let's re-map the Strassen coefficients to these indices.
    
    # M1:
    # A: A00+A11. $i=0,j=0$ and $i=1,j=1$.
    # U[0,0]=1, U[3,0]=1. (Indices 0 and 3)
    # B: B00+B11. $j=0,k=0$ and $j=1,k=1$.
    # V[0,0]=1, V[3,0]=1. (Indices 0 and 3)
    # C: C00+C11. $k=0,i=0$ and $k=1,i=1$.
    # W[0,0]=1, W[3,0]=1. (Indices 0 and 3)
    
    # M2:
    # A: A10+A11. $i=1,j=0$ and $i=1,j=1$.
    # U[2,0]=1, U[3,0]=1. (Indices 2 and 3)
    # B: B00. $j=0,k=0$.
    # V[0,0]=1. (Index 0)
    # C: C10+C11. $k=0,i=1$ and $k=1,i=1$.
    # W[2,0]=1, W[3,0]=1. (Indices 2 and 3)
    
    # M3:
    # A: A00. $i=0,j=0$.
    # U[0,0]=1. (Index 0)
    # B: B10-B11. $j=1,k=0$ and $j=1,k=1$.
    # V[2,0]=1, V[3,0]=-1. (Indices 2 and 3)
    # C: C01+C11. $k=1,i=0$ and $k=1,i=1$.
    # W[1,0]=1, W[3,0]=1. (Indices 1 and 3)
    
    # M4:
    # A: A11. $i=1,j=1$.
    # U[3,0]=1. (Index 3)
    # B: B01-B00. $j=0,k=1$ and $j=0,k=0$.
    # V[1,0]=1, V[0,0]=-1. (Indices 1 and 0)
    # C: C00+C10. $k=0,i=0$ and $k=0,i=1$.
    # W[0,0]=1, W[2,0]=1. (Indices 0 and 2)
    
    # M5:
    # A: A00+A01. $i=0,j=0$ and $i=0,j=1$.
    # U[0,0]=1, U[1,0]=1. (Indices 0 and 1)
    # B: B11. $j=1,k=1$.
    # V[3,0]=1. (Index 3)
    # C: -C00+C01. $k=0,i=0$ and $k=1,i=0$.
    # W[0,0]=-1, W[1,0]=1. (Indices 0 and 1)
    
    # M6:
    # A: A10-A00. $i=1,j=0$ and $i=0,j=0$.
    # U[2,0]=1, U[0,0]=-1. (Indices 2 and 0)
    # B: B00+B01. $j=0,k=0$ and $j=0,k=1$.
    # V[0,0]=1, V[1,0]=1. (Indices 0 and 1)
    # C: C11. $k=1,i=1$.
    # W[3,0]=1. (Index 3)
    
    # M7:
    # A: A01-A11. $i=0,j=1$ and $i=1,j=1$.
    # U[1,0]=1, U[3,0]=-1. (Indices 1 and 3)
    # B: B10+B11. $j=1,k=0$ and $j=1,k=1$.
    # V[2,0]=1, V[3,0]=1. (Indices 2 and 3)
    # C: C00. $k=0,i=0$.
    # W[0,0]=1. (Index 0)
    
    # Now we expand to $6 \times 6 \times 6$.
    # Block indices $I, J, K \in \{0, 1, 2\}$.
    # Local indices $i, j, k \in \{0, 1\}$.
    # Global $i_g = I*2 + i$.
    # Global $j_g = J*2 + j$.
    # Global $k_g = K*2 + k$.
    
    # U global index: $i_g * 6 + j_g$.
    # V global index: $j_g * 6 + k_g$.
    # W global index: $k_g * 6 + i_g$.
    
    # We iterate $I, J, K$ and $r \in 0..6$.
    # For each, we add the local contributions scaled by 1.
    
    strassen_defs = [
        # (U_contribs, V_contribs, W_contribs)
        # contribs is list of (local_index, coeff)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)]),           # M2
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),          # M3
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),          # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),          # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),          # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),          # M7
    ]
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for r in range(7):
                    col = I*7*3 + J*7 + r  # Unique column index
                    u_contribs, v_contribs, w_contribs = strassen_defs[r]
                    
                    for li, coeff in u_contribs:
                        i_g = I*2 + li//2
                        j_g = J*2 + li%2
                        U[i_g*6 + j_g][col] += coeff
                    
                    for lj, coeff in v_contribs:
                        j_g = J*2 + lj//2
                        k_g = K*2 + lj%2
                        V[j_g*6 + k_g][col] += coeff
                    
                    for lk, coeff in w_contribs:
                        k_g = K*2 + lk//2
                        i_g = I*2 + lk%2
                        W[k_g*6 + i_g][col] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
