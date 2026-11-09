import time
import random
import numpy as np

N = 64
NUM = N * N
TOTAL_TIME = 300.0
DEADLINE = TOTAL_TIME - 20

def _build_dist2():
    coords = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
    dx = coords[:, 0][:, np.newaxis] - coords[:, 0][np.newaxis, :]
    dy = coords[:, 1][:, np.newaxis] - coords[:, 1][np.newaxis, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)
    return coords, dist2

def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random()
    coords, dist2 = _build_dist2()
    
    best_S = []
    best_size = 0
    
    def mutual_from_S(S):
        m = {}
        for i in range(len(S)):
            a = S[i]
            for j in range(i + 1, len(S)):
                d = dist2[a, S[j]]
                m[d] = m.get(d, 0) + 1
        return m
    
    def greedy():
        S = []
        in_set = set()
        mutual = {}
        stall = 0
        max_stall = 3000
        while stall < max_stall and time.time() < deadline - 2:
            c = rng.randint(0, NUM - 1)
            if c in in_set:
                stall += 1
                continue
            seen = set()
            valid = True
            for s in S:
                d = dist2[c, s]
                if d in seen or d in mutual:
                    valid = False
                    break
                seen.add(d)
            if not valid:
                stall += 1
                continue
            S.append(c)
            in_set.add(c)
            for s in S[:-1]:
                d = dist2[c, s]
                mutual[d] = mutual.get(d, 0) + 1
            stall = 0
        return S, mutual
    
    def local_search(S, mutual, sub_deadline):
        in_set = set(S)
        bS = list(S)
        bM = dict(mutual)
        bSize = len(S)
        no_improve = 0
        max_no_improve = 30000
        while time.time() < sub_deadline:
            r = rng.random()
            if r < 0.6:
                c = rng.randint(0, NUM - 1)
                if c in in_set:
                    no_improve += 1
                    continue
                seen = set()
                valid = True
                for s in S:
                    d = dist2[c, s]
                    if d in seen or d in mutual:
                        valid = False
                        break
                    seen.add(d)
                if valid:
                    S.append(c)
                    in_set.add(c)
                    for s in S[:-1]:
                        d = dist2[c, s]
                        mutual[d] = mutual.get(d, 0) + 1
                    no_improve = 0
                    if len(S) > bSize:
                        bSize = len(S)
                        bS = list(S)
                        bM = dict(mutual)
                else:
                    no_improve += 1
            elif r < 0.95:
                if len(S) < 2:
                    no_improve += 1
                    continue
                rem = S[rng.randint(0, len(S) - 1)]
                new_S = [s for s in S if s != rem]
                new_M = {}
                for i in range(len(new_S)):
                    a = new_S[i]
                    for j in range(i + 1, len(new_S)):
                        d = dist2[a, new_S[j]]
                        new_M[d] = new_M.get(d, 0) + 1
                found = None
                for _ in range(30):
                    c = rng.randint(0, NUM - 1)
                    if c in in_set and c != rem:
                        continue
                    seen = set()
                    valid = True
                    for s in new_S:
                        d = dist2[c, s]
                        if d in seen or d in new_M:
                            valid = False
                            break
                        seen.add(d)
                    if valid:
                        found = c
                        break
                if found is not None:
                    S = new_S + [found]
                    in_set = set(S)
                    mutual = dict(new_M)
                    for s in new_S:
                        d = dist2[found, s]
                        mutual[d] = mutual.get(d, 0) + 1
                    no_improve = 0
                    if len(S) > bSize:
                        bSize = len(S)
                        bS = list(S)
                        bM = dict(mutual)
                else:
                    no_improve += 1
            else:
                if len(S) < 3:
                    no_improve += 1
                    continue
                rem = S[rng.randint(0, len(S) - 1)]
                S.remove(rem)
                in_set.discard(rem)
                mutual = {}
                for i in range(len(S)):
                    a = S[i]
                    for j in range(i + 1, len(S)):
                        d = dist2[a, S[j]]
                        mutual[d] = mutual.get(d, 0) + 1
                no_improve = 0
            if no_improve > max_no_improve:
                S = list(bS)
                in_set = set(S)
                mutual = dict(bM)
                no_improve = 0
        if len(bS) >= len(S):
            return bS, bM
        return S, mutual
    
    # Phase 1: Greedy restarts
    greedy_end = start + DEADLINE * 0.25
    while time.time() < greedy_end:
        S, mutual = greedy()
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    if not best_S:
        best_S = []
    
    # Phase 2: Local search from best greedy
    mutual_best = mutual_from_S(best_S)
    S2, M2 = local_search(best_S, mutual_best, deadline)
    if len(S2) > best_size:
        best_size = len(S2)
        best_S = S2
    
    # Phase 3: More restarts + local search
    while time.time() < deadline - 3:
        S0, M0 = greedy()
        S1, M1 = local_search(S0, M0, deadline - 3)
        if len(S1) > best_size:
            best_size = len(S1)
            best_S = S1
    
    return [coords[i] for i in best_S]
