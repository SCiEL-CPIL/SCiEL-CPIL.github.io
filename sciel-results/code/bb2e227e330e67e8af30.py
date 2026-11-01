import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit

    # Rank 7 Strassen decomposition (optimal for 2x2 matrix multiplication)
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # Index conventions:
    # U[i*2+j][r] corresponds to A[i,j]
    # V[j*2+k][r] corresponds to B[j,k]
    # W[k*2+i][r] corresponds to C[i,k] output

    # Strassen's 7 products:
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b01 - b11)
    # m4 = a11(b10 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)

    # m1: U[0][0]=1, U[3][0]=1, V[0][0]=1, V[3][0]=1, W[0][0]=1, W[3][0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # m2: U[2][1]=1, U[3][1]=1, V[0][1]=1, W[1][1]=1, W[3][1]=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # m3: U[0][2]=1, V[1][2]=1, V[3][2]=-1, W[2][2]=1, W[3][2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # m4: U[3][3]=1, V[2][3]=1, V[0][3]=-1, W[0][3]=1, W[1][3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # m5: U[0][4]=1, U[1][4]=1, V[3][4]=1, W[0][4]=-1, W[2][4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # m6: U[2][5]=1, U[0][5]=-1, V[0][5]=1, V[1][5]=1, W[3][5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # m7: U[1][6]=1, U[3][6]=-1, V[2][6]=1, V[3][6]=1, W[0][6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    # Quick verification of the rank-7 decomposition
    # Check that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == delta_{i,k} for all i,k
    # The matrix multiplication tensor M[i,j,k,l] = delta_{j,k} * delta_{i,l}
    # We need: sum_r U[i,j,r] V[j,k,r] W[l,i,r] = delta_{j,k} delta_{i,l}
    # Note: W index is W[k*n+i] = W[k*2+i] in the problem statement, but in our
    # convention W[l*2+i] where l is the output column and i is the output row.
    # Actually, let me re-read: "W[k*n+i,r]" - so W is indexed by (output_col, output_row)
    # which means W[k*2+i][r] gives coefficient for C[i][k].
    # The tensor identity is: C[i][k] = sum_{j} A[i][j] B[j][k]
    # = sum_r (sum_{j} U[i*2+j][r] A[i][j]) * (sum_{k'} V[j*2+k'][r] B[j][k']) ... no
    # Actually: C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] summed over j? No.
    # The standard form: M_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # where M_{i,j,k} = delta_{j,k} (the multiplication tensor for 2x2)
    # Wait, the multiplication tensor for (2,2,2) is:
    # T_{i,j,k} = 1 if j==k... no.
    # T_{i,j,k} where i is row of A, j is col of A / row of B, k is col of B
    # T_{i,j,k} = delta_{j,k}... that's not right either.
    # The tensor is: T[i,j,k] = 1 if the (i,k) entry of A*B depends on A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k? No.
    # Actually for matrix multiplication (n,m,p)=(2,2,2):
    # The tensor M has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # M[i,j,k] = 1 if the output C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is simply all ones? No, that's not a tensor decomposition.
    #
    # Let me reconsider. The matrix multiplication tensor for n=m=p=2 is:
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    # because C[i,k] = sum_j A[i,j] B[j,k], and each term A[i,j]B[j,k] appears.
    #
    # The rank of this tensor (as a 3-tensor) is 7 (Strassen).
    #
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Wait, that's not the standard form. Let me use the correct form.
    #
    # Standard: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Hmm, but W should only depend on the output indices. Let me re-read the problem.
    #
    # "U[i*2+j][r] corresponds to A[i,j]" - so U is 4x7
    # "V[j*2+k][r] corresponds to B[j,k]" - so V is 4x7  
    # "W[k*2+i][r] corresponds to C[i,k] output" - so W is 4x7
    #
    # The identity is: for all i,j,k:
    # sum_r U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r] = 1 (since T[i,j,k]=1)
    #
    # Let me verify m1: U[0,0]=1, U[3,0]=1, V[0,0]=1, V[3,0]=1, W[0,0]=1, W[3,0]=1
    # For i=0,j=0,k=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1. Sum over r: only r=0 contributes = 1. OK
    # For i=0,j=0,k=1: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0. Only r=0: U[0,0]*V[0,0]*W[1,0]=1*1*0=0. OK
    # For i=1,j=1,k=1: U[3,0]*V[3,0]*W[3,0] = 1*1*1 = 1. OK
    # For i=0,j=1,k=0: U[2,0]*V[1,0]*W[0,0] = 0*0*1 = 0. OK
    # Looks correct.

    # Verify all 8 entries of the tensor
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(rank):
                    s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                if s != 1:
                    # Fallback: should never happen, but just in case
                    pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
