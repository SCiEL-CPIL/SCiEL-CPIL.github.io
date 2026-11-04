import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
_n, _m, _p = DIMENSIONS

# ---------- exact verification helpers ----------
def _check_decomposition(U, V, W, n=_n, m=_m, p=_p, R=None):
    R = len(U[0]) if R is None else R
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # expected T[i,j,k] = 1 iff i,j,k all valid (always true here)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = Fraction(0)
                for r in range(R):
                    acc += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if acc != 1:
                    return False
    return True

def _random_check(U, V, W, n=_n, m=_m, p=_p, R=None, iters=8, bound=3):
    R = len(U[0]) if R is None else R
    for _ in range(iters):
        A = [[random.randint(-bound, bound) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-bound, bound) for _ in range(p)] for _ in range(m)]
        C = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for r in range(R):
                    acc += U[i*m][r] * V[0*p+k][r] * W[k*n+i][r]  # placeholder, replaced below
                C[i][k] = acc
        # recompute properly (general i,j,k)
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for r in range(R):
                        acc += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] * A[i][j] * B[j][k]
                C[i][k] = acc
        # compare to true product
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    acc += A[i][j]*B[j][k]
                if C[i][k] != acc:
                    return False
    return True

# ---------- Strassen 2x2x2 ----------
def _strassen_2x2():
    # returns U,V,W each 4x7 for 2x2x2, trace W[k*2+i]
    # 7 products:
    # M1=(a11+a22)(b11+b22)
    # M2=(a22)(b21+b22)
    # M3=(a11)(b11)
    # M4=(a22)(b12)
    # M5=(a11+a12)(b22)
    # M6=(a12-a22)(b11+b12)
    # M7=(a11-a12)(b21+b22)
    # c11=M1+M4-M5-M7 ; c12=M3+M5 ; c21=M2+M4 ; c22=M1-M2+M3+M6
    U = [[0]*7 for _ in range(4)]  # idx i*2+j
    V = [[0]*7 for _ in range(4)]  # idx j*2+k
    W = [[0]*7 for _ in range(4)]  # idx k*2+i
    # U row: a11=(0,0)->0, a12=(0,1)->1, a21=(1,0)->2, a22=(1,1)->3
    # V row: b11=(0,0)->0, b12=(0,1)->1, b21=(1,0)->2, b22=(1,1)->3
    # W row: k*2+i ; (k=0,i=0)->0,(k=0,i=1)->1,(k=1,i=0)->2,(k=1,i=1)->3
    # M1
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    # M2
    U[3][1]=1
    V[2][1]=1; V[3][1]=1
    W[2][1]=1; W[3][1]=-1
    # M3
    U[0][2]=1
    V[0][2]=1
    W[0][2]=1
    # M4
    U[3][3]=1
    V[1][3]=1
    W[1][3]=1
    # M5
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[1][4]=-1; W[3][4]=-1
    # M6
    U[1][5]=1; U[3][5]=-1
    V[0][5]=1; V[1][5]=1
    W[0][5]=1
    # M7
    U[0][6]=1; U[1][6]=-1
    V[2][6]=1; V[3][6]=1
    W[2][6]=-1
    return U, V, W

# ---------- bounded-rank exact decomposition of 1 x m x p (row times matrix) ----------
def _row_decompose(m, p, R, seed, deadline):
    """Find U (1 x R), V (m x R), W (p x R) with U[i,j,k]=U0*Vj*Wk=1 for all j,k.
       i.e. U0*V[j][r]*W[k][r] summed over r = 1 for all j,k.
       With U0=1: sum_r V[j][r]*W[k][r] = 1 for all j,k.
       This means (V^T * W) = J (all-ones m x p matrix).
       We need rank <= R such that V^T W = J. J has rank 1, so R>=1 suffices IF
       we can factor J = V^T W with V m x R, W p x R. Since J rank 1, take R=1:
       V = all-ones m x 1, W = all-ones p x 1 -> V^T W = J. But that gives
       U0*V[j]*W[k] = 1*1*1 = 1. So R=1 works! But we must merge with Strassen
       columns. However using R=1 for the row means total rank = 7 + 1 = 8.
       But we must ensure the row's single column doesn't accidentally interact.
       It's independent (separate r index). So R=1 is exact.
    """
    # Trivial exact: R=1, all ones.
    U = [[1]]
    V = [[1] for _ in range(m)]
    W = [[1] for _ in range(p)]
    return U, V, W

def _search_row_rank(m, p, maxR, seed, deadline):
    """Try to find a decomposition of sum_r V[j][r]*W[k][r]=1 with small R.
       R=1 always works. Return best (smallest R) found with exactness.
       We just return R=1 solution (it's optimal since J has rank 1)."""
    U = [[1]]
    V = [[1] for _ in range(m)]
    W = [[1] for _ in range(p)]
    return U, V, W

# ---------- assemble full decomposition ----------
def _assemble(strassen_U, strassen_V, strassen_W, row_U, row_V, row_W, n=_n, m=_m, p=_p):
    # Strassen covers i in {0,1}, j in {0,1}, k in {0,1}
    # Row covers i in {0,1}, j in {2}, k in {0..p-1}
    # We need to cover all (i,j,k). Strassen handles j=0,1; k=0,1.
    # But Strassen only covers k in {0,1}. For j=0,1 and k>=2, we need extra terms.
    # This is a problem: Strassen 2x2x2 only handles a 2x2 A-block and 2x2 B-block.
    # So for j=0,1 we need to handle all k in 0..p-1, not just 0,1.
    # Therefore the simple split is insufficient. We need a different decomposition.
    # Let's instead do: split j into {0,1} and {2}.
    # For j in {0,1}: we have a 2 x 2 x 10 subproblem (A is 2x2, B is 2x10).
    # For j=2: we have a 1 x 1 x 10 subproblem (A is 2x1, B is 1x10).
    # Actually A is 2x3, so rows i=0,1. j indexes columns of A (3 of them).
    # Let me reconsider: A is n x m = 2x3. B is m x p = 3x10.
    # T[i,j,k] = A[i,j]*B[j,k].
    # Decompose by j-groups:
    #   j in {0,1}: A-block is 2x2, B-block is 2x10.
    #   j = 2: A-block is 2x1, B-block is 1x10.
    # For the 2x2 x 10 part, we can use Strassen-like on the 2x2 A with each of the 10 B-columns.
    # That's 7 products per 2 B-columns? No, Strassen is 2x2x2. For 2x2x10, we need
    # to handle B as 2x10. We can tile: 10 = 5 blocks of 2, each using Strassen 7 products
    # = 35 products. Plus the j=2 part: 2x1x10, which is just 10 products (a_i * b_k for each k).
    # Total = 35 + 10 = 45. That's better than 60.
    # But can we do better? 2x2x10: the tensor is 2x2x10. Its rank? 
    # The 2x2x2 part has rank 7. 2x2x10 can be seen as 5 copies of 2x2x2 (if we group k in pairs).
    # Rank of direct sum = sum of ranks = 35. But maybe we can do better by not grouping.
    # Actually 2x2xp tensor: T[i,j,k] = A[i,j]*B[j,k]. This is a "matrix multiplication" tensor
    # for 2x2 * 2xp. The rank is known to be 7*ceil(p/2) for even p? Not exactly.
    # Let me just build the tiling decomposition and verify.
    pass
    return None

def _build_tiled(n=_n, m=_m, p=_p):
    """Build decomposition by tiling j in {0,1} with Strassen over k-pairs,
       and j=2 with schoolbook."""
    U = []  # n*m x rank
    V = []  # m*p x rank
    W = []  # p*n x rank
    # We'll accumulate columns.
    cols_U = []
    cols_V = []
    cols_W = []
    # j in {0,1}, k in pairs (0,1),(2,3),...
    for k0 in range(0, p, 2):
        k1 = k0 + 1
        if k1 >= p:
            break
        sU, sV, sW = _strassen_2x2()
        # sU: 4x7, row = i*2+j for i,j in {0,1}
        # sV: 4x7, row = j*2+k' for j in {0,1}, k' in {0,1}
        # sW: 4x7, row = k'*2+i for k' in {0,1}, i in {0,1}
        for r in range(7):
            cu = [0]*(n*m)
            cv = [0]*(m*p)
            cw = [0]*(p*n)
            for ii in range(2):
                for jj in range(2):
                    cu[ii*m + jj] = sU[ii*2+jj][r]
            for jj in range(2):
                for kk in range(2):
                    cv[jj*p + (k0+kk)] = sV[jj*2+kk][r]
            for kk in range(2):
                for ii in range(2):
                    cw[(k0+kk)*n + ii] = sW[kk*2+ii][r]
            cols_U.append(cu)
            cols_V.append(cv)
            cols_W.append(cw)
    # j = 2, all k
    for k in range(p):
        cu = [0]*(n*m)
        cv = [0]*(m*p)
        cw = [0]*(p*n)
        for ii in range(n):
            cu[ii*m + 2] = 1
        cv[2*p + k] = 1
        cw[k*n + 0] = 1  # i=0
        # Wait, we need W[k*n+i] for each i. For a single product a_i * b_k,
        # T[i,2,k] = a_i * b_k. So U[i*m+2] * V[2*p+k] * W[k*n+i] = 1.
        # With U[i*m+2]=1, V[2*p+k]=1, we need W[k*n+i]=1 for all i.
        for ii in range(n):
            cw[k*n + ii] = 1
        cols_U.append(cu)
        cols_V.append(cv)
        cols_W.append(cw)
    rank = len(cols_U)
    U = cols_U
    V = cols_V
    W = cols_W
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill
    best = None
    best_rank = None

    # Candidate 1: tiled Strassen + schoolbook
    cand = _build_tiled()
    rank = len(cand["U"][0])
    # Verify
    if _check_decomposition(cand["U"], cand["V"], cand["W"]):
        best = cand
        best_rank = rank

    # Try to reduce: maybe some columns are redundant? Check rank of each.
    # For now, return the tiled decomposition.
    if best is None:
        # Fallback: schoolbook
        n, m, p = DIMENSIONS
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return best

# Quick self-test (won't run in evaluator)
if __name__ == "__main__":
    r = construct()
    print("rank:", len(r["U"][0]))
    print("valid:", _check_decomposition(r["U"], r["V"], r["W"]))
