"""Block-partition + Strassen decomposition for 3x5 x 5x5 matrix multiplication.

Design family (kept from proven island): partition the three index sets into
blocks. For each (I-block, J-block, K-block) triple we multiply a sub-matrix of
size (a x b) * (b x c). If a==b==c==2 we use a rank-7 Strassen decomposition;
otherwise we fall back to schoolbook rank a*b*c. The total rank is the sum over
all triples, and the decomposition is exact (integer entries).

We search over block partitions (and their orderings) within a wall-clock budget
and return the best (lowest-rank) exact decomposition found.
"""
import time
import itertools

DIMENSIONS = [3, 5, 5]

_DEADLINE = 150.0
_START = time.time()

# ---------------------------------------------------------------------------
# Strassen rank-7 decomposition for 2x2 * 2x2.
# Returns (U, V, W) each a list of 7 columns; U indexed by (i in 0..1, j in 0..1)
# flattened as i*2+j, V indexed by (j,k) flattened j*2+k, W indexed by (k,i)
# flattened k*2+i.  Entries are integers in {-1,0,1}.
# ---------------------------------------------------------------------------
# Strassen products:
#  M1 = (A00+A11)(B00+B11)
#  M2 = (A10+A11) B00
#  M3 = A00 (B10-B11)
#  M4 = A11 (B01-B10)
#  M5 = (A00+A01) B11
#  M6 = (A10-A00) (B00+B10)
#  M7 = (A01-A11) (B01+B11)
#  C00 = M1 + M4 - M5 - M7
#  C01 = M3 + M5
#  C10 = M2 + M4
#  C11 = M1 - M2 + M3 + M6
#
# U column c: A factors for the c-th Strassen product.
# V column c: B factors for the c-th Strassen product.
# W column c: how M_c contributes to each C entry (k,i).
#
# A is 2x2 with A[i,j], B is 2x2 with B[j,k], C is 2x2 with C[i,k].
# U row index = i*2 + j ; V row index = j*2 + k ; W row index = k*2 + i.

def _strassen222():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    # A part: A00 + A11  -> U[0][0]=1, U[3][0]=1
    # B part: B00 + B11  -> V[0][0]=1, V[3][0]=1
    # C: C00 += M1, C11 += M1  -> W[0*2+0][0]=1 (k=0,i=0), W[1*2+1][0]=1 (k=1,i=1)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1

    # M2 = (A10+A11) B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    # C: C10 += M2, C11 -= M2  -> W[0*2+1][1]=1 (k=0,i=1), W[1*2+1][1]=-1 (k=1,i=1)
    W[1][1] += 1; W[3][1] += -1

    # M3 = A00 (B10-B11)
    U[0][2] += 1
    V[2][2] += 1; V[3][2] += -1
    # C: C01 += M3, C11 += M3  -> W[1*2+0][2]=1 (k=1,i=0), W[1*2+1][2]=1 (k=1,i=1)
    W[2][2] += 1; W[3][2] += 1

    # M4 = A11 (B01-B10)
    U[3][3] += 1
    V[1][3] += 1; V[2][3] += -1
    # C: C00 += M4, C10 += M4  -> W[0*2+0][3]=1 (k=0,i=0), W[0*2+1][3]=1 (k=0,i=1)
    W[0][3] += 1; W[1][3] += 1

    # M5 = (A00+A01) B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    # C: C00 -= M5, C01 += M5  -> W[0*2+0][4]=-1, W[1*2+0][4]=1
    W[0][4] += -1; W[2][4] += 1

    # M6 = (A10-A00) (B00+B10)
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[2][5] += 1
    # C: C11 += M6  -> W[1*2+1][5]=1
    W[3][5] += 1

    # M7 = (A01-A11) (B01+B11)
    U[1][6] += 1; U[3][6] += -1
    V[1][6] += 1; V[3][6] += 1
    # C: C00 -= M7  -> W[0*2+0][6]=-1
    W[0][6] += -1

    return U, V, W

_STR222 = _strassen222()

# ---------------------------------------------------------------------------
# Build a full decomposition from a partition of the three index sets.
# ipart: list of i-block sizes (sum = 3)
# jpart: list of j-block sizes (sum = 5)
# kpart: list of k-block sizes (sum = 5)
# ---------------------------------------------------------------------------
def build(ipart, jpart, kpart):
    n, m, p = DIMENSIONS
    U = [[0]*(n*m) for _ in range(n*m)]   # placeholder, we build columns
    # We'll accumulate columns as lists.
    cols = []
    for a, jstart in _starts(ipart):
        pass
    # Compute start offsets
    i_off = _starts(ipart)
    j_off = _starts(jpart)
    k_off = _starts(kpart)

    Ucols = []
    Vcols = []
    Wcols = []
    for ia, a, ist in zip(range(len(ipart)), ipart, i_off):
        for jb, b, jst in zip(range(len(jpart)), jpart, j_off):
            for kb, c, kst in zip(range(len(kpart)), kpart, k_off):
                # sub-block: a x b * b x c
                if a == 2 and b == 2 and c == 2:
                    sU, sV, sW = _STR222
                    for col in range(7):
                        Ucols.append([sU[r][col] for r in range(4)])
                        Vcols.append([sV[r][col] for r in range(4)])
                        Wcols.append([sW[r][col] for r in range(4)])
                else:
                    # schoolbook a*b*c
                    for ii in range(a):
                        for jj in range(b):
                            for kk in range(c):
                                Ucols.append(_one_at(ist + ii, jst + jj, n*m))
                                Vcols.append(_one_at(jst + jj, kst + kk, m*p))
                                Wcols.append(_one_at(kst + kk, ist + ii, p*n))
    rank = len(Ucols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c in range(rank):
        for r, val in enumerate(Ucols[c]):
            if val:
                U[r][c] = val
        for r, val in enumerate(Vcols[c]):
            if val:
                V[r][c] = val
        for r, val in enumerate(Wcols[c]):
            if val:
                W[r][c] = val
    return U, V, W, rank

def _starts(part):
    out = []
    s = 0
    for sz in part:
        out.append(s)
        s += sz
    return out

def _one_at(i, j, total_rows):
    # a column vector of length total_rows with 1 at position i*m+j style handled by caller
    # here we just need a unit vector; caller passes the flattened row index
    vec = [0]*total_rows
    # i and j already flattened by caller? No: caller passes global row index directly.
    vec[i] = 1
    return vec

# The above _one_at is wrong for flattened (i,j) indices; fix: pass global row index.
def _unit(total_rows, idx):
    vec = [0]*total_rows
    vec[idx] = 1
    return vec

def build2(ipart, jpart, kpart):
    n, m, p = DIMENSIONS
    Ucols = []
    Vcols = []
    Wcols = []
    i_off = _starts(ipart)
    j_off = _starts(jpart)
    k_off = _starts(kpart)
    for a, ist in zip(ipart, i_off):
        for b, jst in zip(jpart, j_off):
            for c, kst in zip(kpart, k_off):
                if a == 2 and b == 2 and c == 2:
                    sU, sV, sW = _STR222
                    for col in range(7):
                        Ucols.append([sU[r][col] for r in range(4)])
                        Vcols.append([sV[r][col] for r in range(4)])
                        Wcols.append([sW[r][col] for r in range(4)])
                else:
                    for ii in range(a):
                        for jj in range(b):
                            for kk in range(c):
                                Ucols.append(_unit(n*m, (ist+ii)*m + (jst+jj)))
                                Vcols.append(_unit(m*p, (jst+jj)*p + (kst+kk)))
                                Wcols.append(_unit(p*n, (kst+kk)*n + (ist+ii)))
    rank = len(Ucols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c in range(rank):
        for r, val in enumerate(Ucols[c]):
            if val:
                U[r][c] = val
        for r, val in enumerate(Vcols[c]):
            if val:
                V[r][c] = val
        for r, val in enumerate(Wcols[c]):
            if val:
                W[r][c] = val
    return U, V, W, rank

# ---------------------------------------------------------------------------
# Verify exactness of a decomposition (tensor equality).
# ---------------------------------------------------------------------------
def verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

# ---------------------------------------------------------------------------
# Enumerate integer compositions of an integer N into parts in a set.
# ---------------------------------------------------------------------------
def compositions(N, max_part):
    # yield tuples of positive ints summing to N, each <= max_part
    def rec(rem, maxp, cur):
        if rem == 0:
            yield tuple(cur)
            return
        for x in range(min(maxp, rem), 0, -1):
            cur.append(x)
            yield from rec(rem - x, maxp, cur)
            cur.pop()
    yield from rec(N, max_part, [])

# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------
def construct():
    global _START
    best = None
    best_rank = None

    # Candidate partitions. i-sums to 3, j and k sum to 5.
    # Parts allowed up to 2 (so we can form 2x2x2 Strassen blocks) plus 1.
    # Actually to get a 2x2x2 block we need a part of size 2 in all three.
    # Allow parts {1,2} for i (sum 3), and {1,2} for j,k (sum 5).
    i_cands = list(compositions(3, 2))
    j_cands = list(compositions(5, 2))
    k_cands = list(compositions(5, 2))

    # Precompute all triples, sort by expected rank (fewer 2x2x2 blocks -> more schoolbook)
    # Expected rank = sum over triples of (7 if (2,2,2) else a*b*c)
    def exp_rank(ip, jp, kp):
        r = 0
        for a in ip:
            for b in jp:
                for c in kp:
                    r += 7 if (a == 2 and b == 2 and c == 2) else a*b*c
        return r

    triples = []
    for ip in i_cands:
        for jp in j_cands:
            for kp in k_cands:
                triples.append((exp_rank(ip, jp, kp), ip, jp, kp))
    triples.sort()

    # Also consider permutations of part orderings? The partition order matters for
    # which indices get grouped, but since all indices are symmetric within a block
    # (the tensor is fully dense with all-ones), the *rank* is the same regardless of
    # ordering. So we only need distinct multisets of part sizes. compositions already
    # yields ordered tuples; we can dedupe by sorted tuple to save time.
    seen = set()
    uniq = []
    for er, ip, jp, kp in triples:
        key = (tuple(sorted(ip)), tuple(sorted(jp)), tuple(sorted(kp)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((er, ip, jp, kp))
    uniq.sort()

    # Evaluate candidates in increasing expected rank; keep the best verified one.
    for er, ip, jp, kp in uniq:
        if time.time() > _DEADLINE - 20:
            break
        # use the canonical (sorted) ordering for the actual build
        ip = tuple(sorted(ip))
        jp = tuple(sorted(jp))
        kp = tuple(sorted(kp))
        U, V, W, rank = build2(ip, jp, kp)
        if not verify(U, V, W):
            continue
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best = (U, V, W)
        # Once we reach the theoretical lower bound we can stop.
        # Lower bound: at least 7 * (number of 2x2x2 blocks) + schoolbook rest.
        if best_rank == er:
            # found a candidate achieving its own expected rank; can't do better than
            # the global minimum expected rank among remaining (they are >= er).
            break

    if best is None:
        # Fallback: schoolbook
        U, V, W, rank = build2((1,1,1), (1,1,1,1,1), (1,1,1,1,1))
        return {"U": U, "V": V, "W": W}

    U, V, W = best
    return {"U": U, "V": V, "W": W}
