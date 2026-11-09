import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 100

    def compute_error(U, V, W, r):
        """Compute max error of the trilinear constraint."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def to_list(M, rows, cols):
        """Convert numpy matrix to list of lists with integer conversion."""
        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                val = M[i, j]
                if abs(val - round(val)) < 1e-10:
                    row.append(int(round(val)))
                else:
                    row.append(val)
            result.append(row)
        return result

    def add_strassen_block(U, V, W, offset, ai, bi, ki):
        """
        Add a Strassen rank-7 decomposition for a 2x2x2 sub-problem.
        A[ai:ai+2, bi:bi+2] * B[bi:bi+2, ki:ki+2] = C[ai:ai+2, ki:ki+2]
        """
        # P1: A[ai,bi] * (B[bi+1,ki+1] - B[bi,ki+1])
        U[ai*m+bi, offset] += 1
        V[(bi+1)*p+(ki+1), offset] += 1
        V[bi*p+(ki+1), offset] += -1
        W[(ki+1)*n+ai, offset] += 1

        # P2: (A[ai,bi] + A[ai,bi+1]) * B[bi,ki+1]
        U[ai*m+bi, offset+1] += 1
        U[ai*m+(bi+1), offset+1] += 1
        V[bi*p+(ki+1), offset+1] += 1
        W[(ki+1)*n+ai, offset+1] += 1

        # P3: (A[ai+1,bi] + A[ai+1,bi+1]) * B[bi,ki]
        U[(ai+1)*m+bi, offset+2] += 1
        U[(ai+1)*m+(bi+1), offset+2] += 1
        V[bi*p+ki, offset+2] += 1
        W[ki*n+(ai+1), offset+2] += 1

        # P4: A[ai+1,bi+1] * (B[bi,ki] - B[bi+1,ki])
        U[(ai+1)*m+(bi+1), offset+3] += 1
        V[bi*p+ki, offset+3] += 1
        V[(bi+1)*p+ki, offset+3] += -1
        W[ki*n+(ai+1), offset+3] += 1

        # P5: (A[ai,bi] + A[ai+1,bi+1]) * (B[bi,ki+1] + B[bi+1,ki+1])
        U[ai*m+bi, offset+4] += 1
        U[(ai+1)*m+(bi+1), offset+4] += 1
        V[bi*p+(ki+1), offset+4] += 1
        V[(bi+1)*p+(ki+1), offset+4] += 1
        W[(ki+1)*n+ai, offset+4] += 1

        # P6: (A[ai,bi+1] - A[ai+1,bi+1]) * (B[bi+1,ki] + B[bi+1,ki+1])
        U[ai*m+(bi+1), offset+5] += 1
        U[(ai+1)*m+(bi+1), offset+5] += -1
        V[(bi+1)*p+ki, offset+5] += 1
        V[(bi+1)*p+(ki+1), offset+5] += 1
        W[ki*n+ai, offset+5] += 1
        W[(ki+1)*n+ai, offset+5] += 1

        # P7: (A[ai,bi] - A[ai+1,bi]) * (B[bi,ki] + B[bi,ki+1])
        U[ai*m+bi, offset+6] += 1
        U[(ai+1)*m+bi, offset+6] += -1
        V[bi*p+ki, offset+6] += 1
        V[bi*p+(ki+1), offset+6] += 1
        W[ki*n+ai, offset+6] += 1
        W[(ki+1)*n+ai, offset+6] += 1

        return offset + 7

    def add_schoolbook_2x2x1(U, V, W, offset, ai, bi, ki):
        """
        Add schoolbook rank-4 decomposition for 2x2x1 sub-problem.
        A[ai:ai+2, bi:bi+2] * B[bi:bi+2, ki] = C[ai:ai+2, ki]
        """
        for di in range(2):
            for dj in range(2):
                U[(ai+di)*m+(bi+dj), offset] += 1
                V[(bi+dj)*p+ki, offset] += 1
                W[ki*n+(ai+di), offset] += 1
                offset += 1
        return offset

    # Build the rank-36 decomposition
    # Split: A is 2x4, split inner dim into 2+2: A = [A1 | A2]
    # B is 4x5, split into B_top (2x5) and B_bottom (2x5)
    # C = A1*B_top + A2*B_bottom
    # For each 2x2 * 2x5: split 5 cols into 2+2+1
    # For the 2x2 blocks: use Strassen (rank 7 each)
    # For the 2x1 block: use schoolbook (rank 4)
    
    # Total: 2 (top/bottom) * (7 + 7 + 4) = 2 * 18 = 36
    
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    offset = 0
    
    # A1 * B_top: A rows 0-1, B inner rows 0-1, output cols 0-4
    # Split output cols: 0-1 (Strassen), 2-3 (Strassen), 4 (schoolbook)
    
    offset = add_strassen_block(U, V, W, offset, 0, 0, 0)
    offset = add_strassen_block(U, V, W, offset, 0, 0, 2)
    offset = add_schoolbook_2x2x1(U, V, W, offset, 0, 0, 4)
    
    # A2 * B_bottom: A rows 0-1, B inner rows 2-3, output cols 0-4
    offset = add_strassen_block(U, V, W, offset, 0, 2, 0)
    offset = add_strassen_block(U, V, W, offset, 0, 2, 2)
    offset = add_schoolbook_2x2x1(U, V, W, offset, 0, 2, 4)
    
    # Verify
    err = compute_error(U, V, W, rank)
    if err < 1e-10:
        return {"U": to_list(U, n*m, rank), "V": to_list(V, m*p, rank), "W": to_list(W, p*n, rank)}
    
    # Fallback to schoolbook rank 40
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    
    return {"U": to_list(U, n*m, rank), "V": to_list(V, m*p, rank), "W": to_list(W, p*n, rank)}
