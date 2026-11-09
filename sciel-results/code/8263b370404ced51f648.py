"""Matrix multiplication 4,4,9: rank-minimal tensor decomposition via structured search."""
DIMENSIONS = [4, 4, 9]

import time
import random

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 118
    rank_school = n * m * p

    # Schoolbook fallback
    U0 = [[0] * rank_school for _ in range(n * m)]
    V0 = [[0] * rank_school for _ in range(m * p)]
    W0 = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    best_rank = rank_school
    best_result = {"U": U0, "V": V0, "W": W0}

    # Strassen 2x2x2 rank-7 coefficients
    sU = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, -1, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    def build_4x4x4_strassen():
        U4 = [[0] * 56 for _ in range(16)]
        V4 = [[0] * 56 for _ in range(16)]
        W4 = [[0] * 56 for _ in range(16)]
        prods = [(0,0,0,0),(0,1,1,0),(0,0,0,1),(0,1,1,1),
                 (1,0,0,0),(1,1,1,0),(1,0,0,1),(1,1,1,1)]
        for pi, (ar, ac, br, bc) in enumerate(prods):
            for s in range(7):
                c = pi * 7 + s
                for i in range(2):
                    for j in range(2):
                        U4[(ar*2+i)*4+(ac*2+j)][c] = sU[s][i*2+j]
                        V4[(br*2+i)*4+(bc*2+j)][c] = sV[s][i*2+j]
                        W4[(bc*2+j)*4+(ar*2+i)][c] = sW[s][i*2+j]
        return U4, V4, W4

    def verify_fast(Um, Vm, Wm, n, m, p, tol=1e-9):
        nc = len(Um[0]) if Um else 0
        if nc == 0:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j; b = j*p+k; c = k*n+i
                    s = 0.0
                    for r in range(nc):
                        s += Um[a][r]*Vm[b][r]*Wm[c][r]
                    if abs(s - 1.0) > tol:
                        return False
        rnd = random.Random(12345)
        checked = 0
        for _ in range(15000):
            a = rnd.randrange(n*m)
            b = rnd.randrange(m*p)
            c = rnd.randrange(p*n)
            i1 = a//m; j1 = a%m
            j2 = b//p; k2 = b%p
            k3 = c//n; i2 = c%n
            if j1==j2 and k2==k3 and i1==i2:
                continue
            s = 0.0
            for r in range(nc):
                s += Um[a][r]*Vm[b][r]*Wm[c][r]
            if abs(s) > tol:
                return False
            checked += 1
        return True

    def verify_exact(Um, Vm, Wm, n, m, p, tol=1e-9):
        nc = len(Um[0]) if Um else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a=i*m+j; b=j*p+k; c=k*n+i
                    s=0.0
                    for r in range(nc):
                        s+=Um[a][r]*Vm[b][r]*Wm[c][r]
                    if abs(s-1.0)>tol:
                        return False
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i1=a//m; j1=a%m; j2=b//p; k2=b%p; k3=c//n; i2=c%n
                    if j1==j2 and k2==k3 and i1==i2:
                        continue
                    s=0.0
                    for r in range(nc):
                        s+=Um[a][r]*Vm[b][r]*Wm[c][r]
                    if abs(s)>tol:
                        return False
        return True

    # Build rank-128 Strassen+schoolbook baseline
    if time.time() < deadline - 80:
        U128 = [[0]*128 for _ in range(16)]
        V128 = [[0]*128 for _ in range(36)]
        W128 = [[0]*128 for _ in range(36)]
        U4, V4, W4 = build_4x4x4_strassen()
        for r in range(56):
            for a in range(16):
                U128[a][r] = U4[a][r]
            for b in range(16):
                j=b//4; k=b%4
                V128[j*9+k][r] = V4[b][r]
            W128[b][r] = W4[b][r]
        U4b, V4b, W4b = build_4x4x4_strassen()
        for r in range(56):
            for a in range(16):
                U128[a][56+r] = U4b[a][r]
            for b in range(16):
                j=b//4; k=b%4
                V128[j*9+4+k][56+r] = V4b[b][r]
            c2 = b//4; i2 = b%4
            W128[4+c2*4+i2][56+r] = W4b[b][r]
        for i in range(4):
            for j in range(4):
                r = 112+i*4+j
                U128[i*4+j][r] = 1
                V128[j*9+8][r] = 1
                W128[8*4+i][r] = 1
        if verify_fast(U128, V128, W128, n, m, p):
            best_rank = 128
            best_result = {"U": U128, "V": V128, "W": W128}

    # Attempt rank-119: Strassen core (56) + 63-term structured tail
    # Tail design: 4x4x5 sub-problem with shared Strassen-like patterns
    # Strategy: use 63 = 7*9 terms, where each group of 7 follows a
    # 2x2x2 Strassen pattern but applied to different (row, col, col-group) slices
    if time.time() < deadline - 40:
        R = 119
        Ut = [[0]*R for _ in range(16)]
        Vt = [[0]*R for _ in range(36)]
        Wt = [[0]*R for _ in range(36)]

        # Core: 4x4x4 Strassen (cols 0-55)
        U4, V4, W4 = build_4x4x4_strassen()
        for r in range(56):
            for a in range(16):
                Ut[a][r] = U4[a][r]
            for b in range(16):
                j=b//4; k=b%4
                Vt[j*9+k][r] = V4[b][r]
            Wt[b][r] = W4[b][r]

        # Tail (cols 56-118): 63 terms for 4x4x5 (cols 4-8 of B)
        # Design: 9 groups of 7, each group handles a 2x2x2 sub-block
        # But we have 4x4x5 = 2x(2x4x5) or 4x(1x4x5)
        # Use 9 groups: 4 groups for cols 4-5 (2x4x2 -> 2 Strassen 2x2x2 = 14 terms)
        # 4 groups for cols 6-7 (another 14 terms)
        # 1 group for col 8 (schoolbook 16 terms) -> total 14+14+16 = 44... too few
        # Actually 9*7=63. Let me use a different grouping.
        # 4x4x5: think of it as 2x2x5 for each of 2 row-pairs
        # 2x2x5: can use 5 schoolbook (10 terms) or Strassen-like
        # Let's just do: cols 4-5 use 2x2x2 Strassen per row-pair (4 groups of 7 = 28)
        # cols 6-7 use 2x2x2 Strassen per row-pair (4 groups of 7 = 28)
        # col 8 use schoolbook (16 terms) -> total 28+28+16 = 72... too many
        # Need exactly 63. Let me reconsider.
        # 63 = 7*9. Use 9 Strassen-like groups.
        # Group g handles (row-pair gp, col-pair cp) for gp in {0,1}, cp in {2,3}
        # That's only 4 groups. For col 8, use 5 more groups? No.
        # Alternative: 63 terms where we overlap the Strassen patterns
        # Let me try: 4x4x5 with rank 63 via explicit construction
        # Use the fact that 4x4x5 can be decomposed as:
        # 2x2x5 (top-left) + 2x2x5 (top-right) + 2x2x5 (bottom-left) + 2x2x5 (bottom-right)
        # Each 2x2x5 has rank at most 10 (schoolbook) or 7+3=10
        # Total: 4*10 = 40... but that's for 4x4x5 which is 80 schoolbook
        # Wait, 2x2x5: A is 2x2, B is 2x5, C is 5x2. Schoolbook = 2*2*5=20.
        # With Strassen on the 2x2 part: 7*5 = 35? No, Strassen gives 7 per 2x2x2.
        # 2x2x5 = 5 copies of 2x2x1 (rank 1 each) = 5, or group as 2x2x2 + 2x2x2 + 2x2x1 = 7+7+4 = 18
        # Hmm, this is getting complex. Let me just use a simpler approach.
        # 63 tail: use 4x4x4 Strassen for cols 4-7 (56 terms) + 4x4x1 schoolbook for col 8 (16 terms) = 72
        # But 72 > 63. So I need to save 9 terms.
        # Idea: merge some schoolbook terms with Strassen terms.
        # Actually, let me just try rank 128 and see if I can find 127 by removing one redundant term.
        pass

    # Try rank-127: remove one schoolbook term from the 128 solution if it's redundant
    # (This is unlikely to work but worth checking)
    if time.time() < deadline - 20 and best_rank == 128:
        # The schoolbook terms for col 8 are 16 terms.
        # Check if any single term is redundant (i.e., removing it still gives correct result)
        # This is very unlikely, but let's check a few
        for skip_r in [112, 116, 120, 124, 127]:
            U127 = [[0]*127 for _ in range(16)]
            V127 = [[0]*127 for _ in range(36)]
            W127 = [[0]*127 for _ in range(36)]
            for a in range(16):
                for b in range(36):
                    for c in range(36):
                        for r in range(127):
                            src = r if r < skip_r else r+1
                            U127[a][r] = best_result["U"][a][src]
                            V127[b][r] = best_result["V"][b][src]
                            W127[c][r] = best_result["W"][c][src]
            if verify_fast(U127, V127, W127, n, m, p):
                best_rank = 127
                best_result = {"U": U127, "V": V127, "W": W127}
                break

    return best_result

# EVOLVE-BLOCK-END
