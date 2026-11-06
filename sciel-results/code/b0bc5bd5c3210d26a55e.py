import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Helper to check if a set of lengths covers 3..39
    def is_pancyclic_lengths(lengths):
        # lengths is a set of ints
        # We need 3..39 to be in lengths. 40 is handled by rim.
        for L in range(3, 40):
            if L not in lengths:
                return False
        return True

    def build_fan_solution(S):
        """
        Build edges and cycles for a fan graph with chords S.
        Returns None if not pancyclic.
        """
        # Check pancyclic
        lengths = {N}
        for k in S:
            lengths.add(k + 1)
            lengths.add(N - k + 1)
        
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                u = S_list[i]
                v = S_list[j]
                lengths.add(v - u + 2)
        
        if not is_pancyclic_lengths(lengths):
            return None
        
        edges = [(i, (i + 1) % N) for i in range(N)]
        for k in S:
            edges.append((0, k))
        
        cycles = []
        # Map lengths to cycle definitions
        # We need to find a valid cycle for each L in 3..N
        # We can reuse the logic from the previous program but ensure it's correct.
        
        length_map = {N: ('rim',)}
        
        for k in S:
            l1 = k + 1
            l2 = N - k + 1
            if l1 not in length_map:
                length_map[l1] = ('chord_short', k)
            if l2 not in length_map:
                length_map[l2] = ('chord_long', k)
        
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                u = S_list[i]
                v = S_list[j]
                l = v - u + 2
                if l not in length_map:
                    length_map[l] = ('chord_pair', u, v)
        
        # Verify all lengths are mapped
        for L in range(3, N + 1):
            if L not in length_map:
                return None # Should not happen if is_pancyclic_lengths passed
        
        for L in range(3, N + 1):
            typ = length_map[L]
            if typ[0] == 'rim':
                cycles.append(list(range(N)))
            elif typ[0] == 'chord_short':
                k = typ[1]
                cycles.append([0] + list(range(1, k + 1)))
            elif typ[0] == 'chord_long':
                k = typ[1]
                cycles.append([0] + list(range(N - 1, k - 1, -1)))
            elif typ[0] == 'chord_pair':
                u, v = typ[1], typ[2]
                cycles.append([0, u] + list(range(u + 1, v + 1)))
        
        return {'edges': edges, 'cycles': cycles}

    def build_two_hub_solution(S0, S20):
        """
        Build edges and cycles for a two-hub graph.
        Hubs at 0 and 20.
        Chords from 0 to S0, from 20 to S20.
        """
        edges = [(i, (i + 1) % N) for i in range(N)]
        for k in S0:
            edges.append((0, k))
        for k in S20:
            edges.append((20, k))
        
        edge_set = set((min(e), max(e)) for e in edges)
        
        # Check if all cycles exist
        # This is harder to verify analytically, so we will construct them.
        # We need to find a cycle of length L for each L in 3..N.
        
        # Strategy:
        # 1. Try rim (40)
        # 2. Try single hub cycles (from 0 or 20)
        # 3. Try pair cycles (within S0, within S20, cross)
        # 4. If any fails, return None.
        
        cycles = [None] * (N + 1)
        
        # Rim
        cycles[N] = list(range(N))
        
        # Single Hub 0
        for k in S0:
            l1 = k + 1
            l2 = N - k + 1
            if cycles[l1] is None:
                cycles[l1] = [0] + list(range(1, k + 1))
            if cycles[l2] is None:
                cycles[l2] = [0] + list(range(N - 1, k - 1, -1))
                
        # Single Hub 20
        for k in S20:
            # Dist from 20 to k
            d1 = (k - 20) % N
            d2 = N - d1
            l1 = d1 + 1
            l2 = d2 + 1
            
            if cycles[l1] is None:
                # Path 20 -> 21 -> ... -> k -> 20
                if d1 > 0:
                    path = [20]
                    cur = 20
                    for _ in range(d1):
                        cur = (cur + 1) % N
                        path.append(cur)
                    cycles[l1] = path
            if cycles[l2] is None:
                # Path 20 -> 19 -> ... -> k -> 20
                if d2 > 0:
                    path = [20]
                    cur = 20
                    for _ in range(d2):
                        cur = (cur - 1) % N
                        path.append(cur)
                    cycles[l2] = path
        
        # Pair Hub 0
        S0_list = sorted(S0)
        for i in range(len(S0_list)):
            for j in range(i + 1, len(S0_list)):
                u = S0_list[i]
                v = S0_list[j]
                l = v - u + 2
                if cycles[l] is None:
                    cycles[l] = [0, u] + list(range(u + 1, v + 1))
                    
        # Pair Hub 20
        S20_list = sorted(S20)
        for i in range(len(S20_list)):
            for j in range(i + 1, len(S20_list)):
                u = S20_list[i]
                v = S20_list[j]
                # Cycle 20 -> u -> ... -> v -> 20
                # Path u to v on rim
                d = (v - u) % N
                if d == 0: continue
                l = d + 2
                if cycles[l] is None:
                    path = [20]
                    cur = 20
                    # Go to u
                    du = (u - 20) % N
                    for _ in range(du):
                        cur = (cur + 1) % N
                        path.append(cur)
                    # Go to v
                    for _ in range(d):
                        cur = (cur + 1) % N
                        path.append(cur)
                    cycles[l] = path
        
        # Cross Pairs
        for u in S0:
            for v in S20:
                # Cycle 0 -> u -> ... -> v -> 20 -> ... -> 0
                # Path u to v
                d1 = (v - u) % N
                d2 = N - d1
                
                # Option 1: u -> ... -> v (clockwise)
                l1 = d1 + 1 + 1 + 20 # 1 (0-u) + d1 (u-v) + 1 (v-20) + 20 (20-0)
                if cycles[l1] is None:
                    path = [0]
                    cur = 0
                    # Go to u
                    du = (u - 0) % N
                    for _ in range(du):
                        cur = (cur + 1) % N
                        path.append(cur)
                    # Go to v
                    for _ in range(d1):
                        cur = (cur + 1) % N
                        path.append(cur)
                    # Go to 20
                    dv = (20 - v) % N
                    for _ in range(dv):
                        cur = (cur + 1) % N
                        path.append(cur)
                    # Go to 0
                    d0 = (0 - 20) % N
                    for _ in range(d0):
                        cur = (cur + 1) % N
                        path.append(cur)
                    if len(path) == l1:
                        cycles[l1] = path
                        
                # Option 2: u -> ... -> v (counter-clockwise)
                l2 = d2 + 1 + 1 + 20
                if cycles[l2] is None:
                    path = [0]
                    cur = 0
                    # Go to u
                    du = (u - 0) % N
                    for _ in range(du):
                        cur = (cur + 1) % N
                        path.append(cur)
                    # Go to v (counter)
                    for _ in range(d2):
                        cur = (cur - 1) % N
                        path.append(cur)
                    # Go to 20
                    dv = (20 - v) % N
                    for _ in range(dv):
                        cur = (cur + 1) % N
                        path.append(cur)
                    # Go to 0
                    d0 = (0 - 20) % N
                    for _ in range(d0):
                        cur = (cur + 1) % N
                        path.append(cur)
                    if len(path) == l2:
                        cycles[l2] = path
        
        # Verify all cycles are found and valid
        for L in range(3, N + 1):
            if cycles[L] is None:
                return None
            # Verify edge existence
            cyc = cycles[L]
            for i in range(len(cyc)):
                a = cyc[i]
                b = cyc[(i + 1) % len(cyc)]
                if (min(a, b), max(a, b)) not in edge_set:
                    return None
            if len(set(cyc)) != len(cyc):
                return None
        
        return {'edges': edges, 'cycles': cycles}

    # Search for 7-chord Two-Hub solution
    def search_two_hub(m):
        candidates = list(range(2, N - 1))
        # Exclude 20 from hub 20 candidates? 20 is the hub.
        # Chords from 0: 2..38
        # Chords from 20: 2..38 (excluding 20? No, 20 is hub, chord is 20-k)
        
        for restart in range(2000):
            if time.time() > deadline:
                break
            
            # Randomly split m into m0, m20
            m0 = random.randint(max(1, m // 2 - 2), min(m - 1, m // 2 + 2))
            m20 = m - m0
            
            S0 = set(random.sample(candidates, m0))
            S20 = set(random.sample([x for x in candidates if x not in S0], m20))
            
            # Local search
            best_S0 = set(S0)
            best_S20 = set(S20)
            
            for step in range(50):
                if time.time() > deadline:
                    break
                
                # Try swapping
                if random.random() < 0.5:
                    # Swap in S0
                    if len(S0) > 0 and len([x for x in candidates if x not in S0 and x not in S20]) > 0:
                        u = random.choice(list(S0))
                        v = random.choice([x for x in candidates if x not in S0 and x not in S20])
                        new_S0 = set(S0)
                        new_S0.remove(u)
                        new_S0.add(v)
                        res = build_two_hub_solution(new_S0, S20)
                        if res:
                            return res
                else:
                    # Swap in S20
                    if len(S20) > 0 and len([x for x in candidates if x not in S0 and x not in S20]) > 0:
                        u = random.choice(list(S20))
                        v = random.choice([x for x in candidates if x not in S0 and x not in S20])
                        new_S20 = set(S20)
                        new_S20.remove(u)
                        new_S20.add(v)
                        res = build_two_hub_solution(S0, new_S20)
                        if res:
                            return res
            
            # Try greedy improvement?
            # For now, just random restarts.
            
    # Try 7 chords
    res = search_two_hub(7)
    if res:
        return res

    # Fallback to 8-chord Fan
    def search_fan(m):
        candidates = list(range(2, N - 1))
        
        for restart in range(5000):
            if time.time() > deadline:
                break
            
            S = set()
            if m >= 2:
                S.add(2)
                S.add(38)
            remaining = [k for k in candidates if k not in S]
            if len(remaining) >= m - len(S):
                S.update(random.sample(remaining, m - len(S)))
            
            # Local search
            for step in range(100):
                if time.time() > deadline:
                    break
                
                in_S = list(S)
                out_S = [k for k in candidates if k not in S]
                
                if not in_S or not out_S:
                    break
                
                u = random.choice(in_S)
                v = random.choice(out_S)
                
                new_S = set(S)
                new_S.remove(u)
                new_S.add(v)
                
                res = build_fan_solution(new_S)
                if res:
                    return res
                
                # Accept neutral moves?
                # We don't have a score, so we just check validity.
                # To escape local minima, we might need to accept worse "coverage" but we don't track it here.
                # Simple random walk.
                S = new_S
                
    res = search_fan(8)
    if res:
        return res

    # Final fallback: 9 chords
    res = search_fan(9)
    if res:
        return res

    # Should not reach here
    raise Exception("Failed to construct pancyclic graph")

