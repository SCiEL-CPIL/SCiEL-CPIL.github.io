<<<<<<< SEARCH
def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strategy: Use block decomposition with multiple splits and pick the best rank.
    # We'll try several different block splits and return the one with lowest rank.
    
    candidates = []
    
    # Split 1: n=[1,2], m=[2,3], p=[2,3]
    # Blocks: (1,2,2)=4, (1,3,3)=9, (2,2,2)=7, (2,3,3)=18 -> total 38
    candidates.append(([1, 2], [2, 3], [2, 3]))
    
    # Split 2: n=[1,2], m=[3,2], p=[3,2]
    # Blocks: (1,3,3)=9, (1,2,2)=4, (2,3,3)=18, (2,2,2)=7 -> total 38
    candidates.append(([1, 2], [3, 2], [3, 2]))
    
    # Split 3: n=[2,1], m=[2,3], p=[2,3]
    # Blocks: (2,2,2)=7, (2,3,3)=18, (1,2,2)=4, (1,3,3)=9 -> total 38
    candidates.append(([2, 1], [2, 3], [2, 3]))
    
    # Split 4: n=[1,1,1], m=[2,3], p=[2,3]
    # 6 blocks: (1,2,2)=4, (1,3,3)=9, (1,2,2)=4, (1,3,3)=9, (1,2,2)=4, (1,3,3)=9 -> total 39
    candidates.append(([1, 1, 1], [2, 3], [2, 3]))
    
    # Split 6: n=[1,2], m=[2,1,2], p=[2,1,2]
    # i=0: (1,2,2)=4, (1,1,1)=1, (1,2,2)=4
    # i=1: (2,2,2)=7, (2,1,1)=2, (2,2,2)=7
    # Total: 4+1+4+7+2+7 = 25
    candidates.append(([1, 2], [2, 1, 2], [2, 1, 2]))
    
    # Split 7: n=[1,2], m=[1,2,2], p=[1,2,2]
    # i=0: (1,1,1)=1, (1,2,2)=4, (1,2,2)=4
    # i=1: (2,1,1)=2, (2,2,2)=7, (2,2,2)=7
    # Total: 1+4+4+2+7+7 = 25
    candidates.append(([1, 2], [1, 2, 2], [1, 2, 2]))
    
    # Split 8: n=[1,1,1], m=[1,2,2], p=[1,2,2]
    # 9 blocks: 1+4+4 + 1+4+4 + 1+4+4 = 27
    candidates.append(([1, 1, 1], [1, 2, 2], [1, 2, 2]))
    
    # Split 9: n=[1,2], m=[2,2,1], p=[2,2,1]
    # i=0: (1,2,2)=4, (1,2,2)=4, (1,1,1)=1
    # i=1: (2,2,2)=7, (2,2,2)=7, (2,1,1)=2
    # Total: 4+4+1+7+7+2 = 25
    candidates.append(([1, 2], [2, 2, 1], [2, 2, 1]))
    
    # Split 10: n=[1,2], m=[1,1,3], p=[1,1,3]
    # i=0: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # i=1: (2,1,1)=2, (2,1,1)=2, (2,3,3)=18
    # Total: 1+1+9+2+2+18 = 33
    candidates.append(([1, 2], [1, 1, 3], [1, 1, 3]))
    
    # New splits to explore lower rank:
    
    # Split 11: n=[1,1,1], m=[2,2,1], p=[2,2,1]
    # 9 blocks: each row of n has blocks (1,2,2)=4, (1,2,2)=4, (1,1,1)=1
    # Total: 3*(4+4+1) = 27
    candidates.append(([1, 1, 1], [2, 2, 1], [2, 2, 1]))
    
    # Split 12: n=[1,1,1], m=[1,2,2], p=[2,2,1]
    # i=0: (1,1,2)=2, (1,2,2)=4, (1,2,1)=2
    # i=1: (1,1,2)=2, (1,2,2)=4, (1,2,1)=2
    # i=2: (1,1,2)=2, (1,2,2)=4, (1,2,1)=2
    # Total: 3*(2+4+2) = 24
    candidates.append(([1, 1, 1], [1, 2, 2], [2, 2, 1]))
    
    # Split 13: n=[1,1,1], m=[2,2,1], p=[1,2,2]
    # i=0: (1,2,1)=2, (1,2,2)=4, (1,1,2)=2
    # i=1: (1,2,1)=2, (1,2,2)=4, (1,1,2)=2
    # i=2: (1,2,1)=2, (1,2,2)=4, (1,1,2)=2
    # Total: 3*(2+4+2) = 24
    candidates.append(([1, 1, 1], [2, 2, 1], [1, 2, 2]))
    
    # Split 14: n=[1,1,1], m=[1,1,3], p=[1,1,3]
    # i=0: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # i=1: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # i=2: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # Total: 3*(1+1+9) = 33
    candidates.append(([1, 1, 1], [1, 1, 3], [1, 1, 3]))
    
    # Split 15: n=[1,2], m=[2,1,2], p=[1,2,2]
    # i=0: (1,2,1)=2, (1,1,2)=2, (1,2,2)=4
    # i=1: (2,2,1)=4, (2,1,2)=4, (2,2,2)=7
    # Total: 2+2+4+4+4+7 = 23
    candidates.append(([1, 2], [2, 1, 2], [1, 2, 2]))
    
    # Split 16: n=[1,2], m=[1,2,2], p=[2,1,2]
    # i=0: (1,1,2)=2, (1,2,1)=2, (1,2,2)=4
    # i=1: (2,1,2)=4, (2,2,1)=4, (2,2,2)=7
    # Total: 2+2+4+4+4+7 = 23
    candidates.append(([1, 2], [1, 2, 2], [2, 1, 2]))
    
    # Split 17: n=[2,1], m=[2,1,2], p=[1,2,2]
    # i=0: (2,2,1)=4, (2,1,2)=4, (2,2,2)=7
    # i=1: (1,2,1)=2, (1,1,2)=2, (1,2,2)=4
    # Total: 4+4+7+2+2+4 = 23
    candidates.append(([2, 1], [2, 1, 2], [1, 2, 2]))
    
    # Split 18: n=[2,1], m=[1,2,2], p=[2,1,2]
    # i=0: (2,1,2)=4, (2,2,1)=4, (2,2,2)=7
    # i=1: (1,1,2)=2, (1,2,1)=2, (1,2,2)=4
    # Total: 4+4+7+2+2+4 = 23
    candidates.append(([2, 1], [1, 2, 2], [2, 1, 2]))
    
    # Split 19: n=[1,1,1], m=[2,1,2], p=[2,1,2]
    # 9 blocks: each (1,2,2)=4, (1,1,1)=1, (1,2,2)=4
    # Total: 3*(4+1+4) = 27
    candidates.append(([1, 1, 1], [2, 1, 2], [2, 1, 2]))
    
    # Split 20: n=[1,2], m=[2,2,1], p=[1,2,2]
    # i=0: (1,2,1)=2, (1,2,2)=4, (1,1,2)=2
    # i=1: (2,2,1)=4, (2,2,2)=7, (2,1,2)=4
    # Total: 2+4+2+4+7+4 = 23
    candidates.append(([1, 2], [2, 2, 1], [1, 2, 2]))
    
    # Split 21: n=[1,2], m=[1,2,2], p=[2,2,1]
    # i=0: (1,1,2)=2, (1,2,2)=4, (1,2,1)=2
    # i=1: (2,1,2)=4, (2,2,2)=7, (2,2,1)=4
    # Total: 2+4+2+4+7+4 = 23
    candidates.append(([1, 2], [1, 2, 2], [2, 2, 1]))
    
    # Split 22: n=[2,1], m=[2,2,1], p=[1,2,2]
    # i=0: (2,2,1)=4, (2,2,2)=7, (2,1,2)=4
    # i=1: (1,2,1)=2, (1,2,2)=4, (1,1,2)=2
    # Total: 4+7+4+2+4+2 = 23
    candidates.append(([2, 1], [2, 2, 1], [1, 2, 2]))
    
    # Split 23: n=[2,1], m=[1,2,2], p=[2,2,1]
    # i=0: (2,1,2)=4, (2,2,2)=7, (2,2,1)=4
    # i=1: (1,1,2)=2, (1,2,2)=4, (1,2,1)=2
    # Total: 4+7+4+2+4+2 = 23
    candidates.append(([2, 1], [1, 2, 2], [2, 2, 1]))
    
    # Split 24: n=[1,1,1], m=[1,2,2], p=[1,2,2]
    # Already covered by split 8, skip
    
    # Split 25: n=[1,2], m=[1,1,3], p=[1,2,2]
    # i=0: (1,1,1)=1, (1,1,2)=2, (1,3,2)=6
    # i=1: (2,1,1)=2, (2,1,2)=4, (2,3,2)=12
    # Total: 1+2+6+2+4+12 = 27
    candidates.append(([1, 2], [1, 1, 3], [1, 2, 2]))
    
    # Split 26: n=[1,2], m=[1,2,2], p=[1,1,3]
    # i=0: (1,1,1)=1, (1,2,1)=2, (1,2,3)=6
    # i=1: (2,1,1)=2, (2,2,1)=4, (2,2,3)=12
    # Total: 1+2+6+2+4+12 = 27
    candidates.append(([1, 2], [1, 2, 2], [1, 1, 3]))
    
    # Split 27: n=[1,1,1], m=[1,1,3], p=[1,2,2]
    # i=0: (1,1,1)=1, (1,1,2)=2, (1,3,2)=6
    # i=1: (1,1,1)=1, (1,1,2)=2, (1,3,2)=6
    # i=2: (1,1,1)=1, (1,1,2)=2, (1,3,2)=6
    # Total: 3*(1+2+6) = 27
    candidates.append(([1, 1, 1], [1, 1, 3], [1, 2, 2]))
    
    # Split 28: n=[1,1,1], m=[1,2,2], p=[1,1,3]
    # i=0: (1,1,1)=1, (1,2,1)=2, (1,2,3)=6
    # i=1: (1,1,1)=1, (1,2,1)=2, (1,2,3)=6
    # i=2: (1,1,1)=1, (1,2,1)=2, (1,2,3)=6
    # Total: 3*(1+2+6) = 27
    candidates.append(([1, 1, 1], [1, 2, 2], [1, 1, 3]))
    
    # Split 29: n=[1,2], m=[2,1,2], p=[2,2,1]
    # i=0: (1,2,2)=4, (1,1,2)=2, (1,2,1)=2
    # i=1: (2,2,2)=7, (2,1,2)=4, (2,2,1)=4
    # Total: 4+2+2+7+4+4 = 23
    candidates.append(([1, 2], [2, 1, 2], [2, 2, 1]))
    
    # Split 30: n=[1,2], m=[1,2,2], p=[1,2,2]
    # Already covered by split 7, skip
    
    # Split 31: n=[2,1], m=[2,1,2], p=[2,2,1]
    # i=0: (2,2,2)=7, (2,1,2)=4, (2,2,1)=4
    # i=1: (1,2,2)=4, (1,1,2)=2, (1,2,1)=2
    # Total: 7+4+4+4+2+2 = 23
    candidates.append(([2, 1], [2, 1, 2], [2, 2, 1]))
    
    # Split 32: n=[2,1], m=[1,2,2], p=[1,2,2]
    # i=0: (2,1,1)=2, (2,2,2)=7, (2,2,2)=7
    # i=1: (1,1,1)=1, (1,2,2)=4, (1,2,2)=4
    # Total: 2+7+7+1+4+4 = 25
    candidates.append(([2, 1], [1, 2, 2], [1, 2, 2]))
    
    # Split 33: n=[1,2], m=[1,2,2], p=[2,1,2]
    # Already covered by split 16, skip
    
    # Split 34: n=[1,2], m=[2,2,1], p=[2,1,2]
    # i=0: (1,2,2)=4, (1,2,1)=2, (1,1,2)=2
    # i=1: (2,2,2)=7, (2,2,1)=4, (2,1,2)=4
    # Total: 4+2+2+7+4+4 = 23
    candidates.append(([1, 2], [2, 2, 1], [2, 1, 2]))
    
    # Split 35: n=[2,1], m=[2,2,1], p=[2,1,2]
    # i=0: (2,2,2)=7, (2,2,1)=4, (2,1,2)=4
    # i=1: (1,2,2)=4, (1,2,1)=2, (1,1,2)=2
    # Total: 7+4+4+4+2+2 = 23
    candidates.append(([2, 1], [2, 2, 1], [2, 1, 2]))
    
    # Split 36: n=[1,1,1], m=[2,2,1], p=[1,1,3]
    # i=0: (1,2,1)=2, (1,2,1)=2, (1,1,3)=3
    # i=1: (1,2,1)=2, (1,2,1)=2, (1,1,3)=3
    # i=2: (1,2,1)=2, (1,2,1)=2, (1,1,3)=3
    # Total: 3*(2+2+3) = 21
    candidates.append(([1, 1, 1], [2, 2, 1], [1, 1, 3]))
    
    # Split 37: n=[1,1,1], m=[1,1,3], p=[2,2,1]
    # i=0: (1,1,2)=2, (1,1,2)=2, (1,3,1)=3
    # i=1: (1,1,2)=2, (1,1,2)=2, (1,3,1)=3
    # i=2: (1,1,2)=2, (1,1,2)=2, (1,3,1)=3
    # Total: 3*(2+2+3) = 21
    candidates.append(([1, 1, 1], [1, 1, 3], [2, 2, 1]))
    
    # Split 38: n=[1,1,1], m=[2,1,2], p=[1,1,3]
    # i=0: (1,2,1)=2, (1,1,1)=1, (1,2,3)=6
    # i=1: (1,2,1)=2, (1,1,1)=1, (1,2,3)=6
    # i=2: (1,2,1)=2, (1,1,1)=1, (1,2,3)=6
    # Total: 3*(2+1+6) = 27
    candidates.append(([1, 1, 1], [2, 1, 2], [1, 1, 3]))
    
    # Split 39: n=[1,1,1], m=[1,1,3], p=[2,1,2]
    # i=0: (1,1,2)=2, (1,1,1)=1, (1,3,2)=6
    # i=1: (1,1,2)=2, (1,1,1)=1, (1,3,2)=6
    # i=2: (1,1,2)=2, (1,1,1)=1, (1,3,2)=6
    # Total: 3*(2+1+6) = 27
    candidates.append(([1, 1, 1], [1, 1, 3], [2, 1, 2]))
    
    # Split 40: n=[1,2], m=[2,1,2], p=[1,1,3]
    # i=0: (1,2,1)=2, (1,1,1)=1, (1,2,3)=6
    # i=1: (2,2,1)=4, (2,1,1)=2, (2,2,3)=12
    # Total: 2+1+6+4+2+12 = 27
    candidates.append(([1, 2], [2, 1, 2], [1, 1, 3]))
    
    # Split 41: n=[1,2], m=[1,1,3], p=[2,1,2]
    # i=0: (1,1,2)=2, (1,1,1)=1, (1,3,2)=6
    # i=1: (2,1,2)=4, (2,1,1)=2, (2,3,2)=12
    # Total: 2+1+6+4+2+12 = 27
    candidates.append(([1, 2], [1, 1, 3], [2, 1, 2]))
    
    # Split 42: n=[2,1], m=[2,1,2], p=[1,1,3]
    # i=0: (2,2,1)=4, (2,1,1)=2, (2,2,3)=12
    # i=1: (1,2,1)=2, (1,1,1)=1, (1,2,3)=6
    # Total: 4+2+12+2+1+6 = 27
    candidates.append(([2, 1], [2, 1, 2], [1, 1, 3]))
    
    # Split 43: n=[2,1], m=[1,1,3], p=[2,1,2]
    # i=0: (2,1,2)=4, (2,1,1)=2, (2,3,2)=12
    # i=1: (1,1,2)=2, (1,1,1)=1, (1,3,2)=6
    # Total: 4+2+12+2+1+6 = 27
    candidates.append(([2, 1], [1, 1, 3], [2, 1, 2]))
    
    # Split 44: n=[1,1,1], m=[1,2,2], p=[1,1,3]
    # Already covered by split 28, skip
    
    # Split 45: n=[1,1,1], m=[1,1,3], p=[1,2,2]
    # Already covered by split 27, skip
    
    # Split 46: n=[1,2], m=[1,1,3], p=[1,1,3]
    # i=0: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # i=1: (2,1,1)=2, (2,1,1)=2, (2,3,3)=18
    # Total: 1+1+9+2+2+18 = 33
    candidates.append(([1, 2], [1, 1, 3], [1, 1, 3]))
    
    # Split 47: n=[2,1], m=[1,1,3], p=[1,1,3]
    # i=0: (2,1,1)=2, (2,1,1)=2, (2,3,3)=18
    # i=1: (1,1,1)=1, (1,1,1)=1, (1,3,3)=9
    # Total: 2+2+18+1+1+9 = 33
    candidates.append(([2, 1], [1, 1, 3], [1, 1, 3]))
    
    # Split 48: n=[1,1,1], m=[2,2,1], p=[2,1,2]
    # i=0: (1,2,2)=4, (1,2,1)=2, (1,1,2)=2
    # i=1: (1,2,2)=4, (1,2,1)=2, (1,1,2)=2
    # i=2: (1,2,2)=4, (1,2,1)=2, (1,1,2)=2
    # Total: 3*(4+2+2) = 24
    candidates.append(([1, 1, 1], [2, 2, 1], [2, 1, 2]))
    
    # Split 49: n=[1,1,1], m=[1,2,2], p=[2,2,1]
    # Already covered by split 12, skip
    
    # Split 50: n=[1,1,1], m=[2,1,2], p=[2,2,1]
    # i=0: (1,2,2)=4, (1,1,2)=2, (1,2,1)=2
    # i=1: (1,2,2)=4, (1,1,2)=2, (1,2,1)=2
    # i=2: (1,2,2)=4, (1,1,2)=2, (1,2,1)=2
    # Total: 3*(4+2+2) = 24
    candidates.append(([1, 1, 1], [2, 1, 2], [2, 2, 1]))
    
    # Split 51: n=[1,1,1], m=[1,2,2], p=[1,2,2]
    # Already covered by split 8, skip
    
    # Split 52: n=[1,1,1], m=[2,2,1], p=[1,2,2]
    # Already covered by split 13, skip
    
    # Split 53: n=[1,1,1], m=[1,1,3], p=[1,1,3]
    # Already covered by split 14, skip
    
    # Split 54: n=[1,1,1], m=[1,1,3], p=[2,1,2]
    # Already covered by split 39, skip
    
    # Split 55: n=[1,1,1], m=[2,1,2], p=[1,2,2]
    # i=0: (1,2,1)=2, (1,1,2)=2, (1,2,2)=4
    # i=1: (1,2,1)=2, (1,1,2)=2, (1,2,2)=4
    # i=2: (1,2,1)=2, (1,1,2)=2, (1,2,2)=4
    # Total: 3*(2+2+4) = 24
    candidates.append(([1, 1, 1], [2, 1, 2], [1, 2, 2]))
    
    # Split 56: n=[1,1,1], m=[1,2,2], p=[2,1,2]
    # i=0: (1,1,2)=2, (1,2,1)=2, (1,2,2)=4
    # i=1: (1,1,2)=2, (1,2,1)=2, (1,2,2)=4
    # i=2: (1,1,2)=2, (1,2,1)=2, (1,2,2)=4
    # Total: 3*(2+2+4) = 24
    candidates.append(([1, 1, 1], [1, 2, 2], [2, 1, 2]))
    
    # Split 57: n=[1,1,1], m=[1,1,3], p=[1,2,2]
    # Already covered by split 27, skip
    
    # Split 58: n=[1,1,1], m=[1,2,2], p=[1,1,3]
    # Already covered by split 28, skip
    
    # Try all candidates and pick the best
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Sort candidates by estimated rank (lower first) to find good solutions early
    def estimate_rank(n_parts, m_parts, p_parts):
        total = 0
        for ni in n_parts:
            for mi in m_parts:
                for pi in p_parts:
                    if ni == 2 and mi == 2 and pi == 2:
                        total += 7
                    else:
                        total += ni * mi * pi
        return total
    
    sorted_candidates = sorted(candidates, key=lambda x: estimate_rank(x[0], x[1], x[2]))
    
    for n_parts, m_parts, p_parts in sorted_candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U, V, W
        except Exception:
            continue
    
    # Fallback to schoolbook if something went wrong
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    np.random.seed(42)
    for _ in range(30):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        expected = 1.0
        actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        if abs(expected - actual) > 1e-9:
            return False
    return True


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strategy: Systematically explore 5=2+2+1 block decompositions
    # All permutations of [2,2,1] for each dimension give 3x3x3=27 candidates
    # Plus some 2-way splits for safety
    
    import itertools
    
    # Distinct permutations of [2,2,1]: (2,2,1), (2,1,2), (1,2,2)
    perms = [(2,2,1), (2,1,2), (1,2,2)]
    
    candidates = []
    seen = set()
    
    # All 27-block decompositions with 5=2+2+1
    for np_parts in perms:
        for mp_parts in perms:
            for pp_parts in perms:
                key = (np_parts, mp_parts, pp_parts)
                if key not in seen:
                    seen.add(key)
                    candidates.append((list(np_parts), list(mp_parts), list(pp_parts)))
    
    # Also add 2-way splits as fallback
    two_way = [(1,2), (2,1)]
    for np_parts in two_way:
        for mp_parts in two_way:
            for pp_parts in two_way:
                key = (np_parts, mp_parts, pp_parts)
                if key not in seen:
                    seen.add(key)
                    candidates.append((list(np_parts), list(mp_parts), list(pp_parts)))
    
    # Estimate rank and sort
    def estimate_rank(n_parts, m_parts, p_parts):
        total = 0
        for ni in n_parts:
            for mi in m_parts:
                for pi in p_parts:
                    if ni == 2 and mi == 2 and pi == 2:
                        total += 7
                    else:
                        total += ni * mi * pi
        return total
    
    sorted_candidates = sorted(candidates, key=lambda x: estimate_rank(x[0], x[1], x[2]))
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    for n_parts, m_parts, p_parts in sorted_candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            if rank < best_rank:
                # Verify correctness before accepting
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
        except Exception:
            continue
    
    # Fallback to schoolbook if something went wrong
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
