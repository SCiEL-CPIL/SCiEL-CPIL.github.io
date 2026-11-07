"""Combined program: precomputed comparability matrix + multi-restart local search with chain decomposition."""
import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 82
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute comparability matrix: comp[a][b] = 1 if a is strict subset of b or b is strict subset of a
    # Using a flat array for cache efficiency: comp[a*N + b]
    comp = bytearray(N * N)
    for a in range(N):
        for b in range(N):
            if a != b and (a | b) == a:
                comp[a * N + b] = 1
                comp[b * N + a] = 1
    
    # Precompute popcount
    popcount = [0] * N
    for i in range(1, N):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    def compute_score(fam):
        """Compute total comparable pairs in family."""
        cnt = 0
        for i in range(len(fam)):
            base_i = fam[i] * N
            for j in range(i + 1, len(fam)):
                if comp[base_i + fam[j]]:
                    cnt += 1
        return cnt
    
    def get_degree(mask, fam, exclude_idx):
        """Count comparable sets with mask in fam, excluding position exclude_idx."""
        base = mask * N
        cnt = 0
        for i, s in enumerate(fam):
            if i != exclude_idx and comp[base + s]:
                cnt += 1
        return cnt
    
    def make_chain_family(rng):
        """Create initial family using chain decomposition with greedy fill."""
        family = []
        seen = set()
        
        # Generate multiple chains with different orderings
        base_order = list(range(n))
        chains = []
        
        # Chain 1: natural order
        chains.append(list(base_order))
        
        # Chain 2: reversed
        chains.append(list(reversed(base_order)))
        
        # Chain 3: rotate by 1
        chains.append(base_order[1:] + base_order[:1])
        
        # Chain 4: rotate by 2
        chains.append(base_order[2:] + base_order[:2])
        
        # Chain 5: random shuffle
        shuffled = list(base_order)
        rng.shuffle(shuffled)
        chains.append(shuffled)
        
        # Chain 6: another random shuffle
        shuffled2 = list(base_order)
        rng.shuffle(shuffled2)
        chains.append(shuffled2)
        
        # Build family from chains
        for chain in chains:
            s = 0
            seen.add(s)
            if s not in seen:
                family.append(s)
            for v in chain:
                s |= (1 << v)
                if s not in seen:
                    family.append(s)
                    seen.add(s)
        
        # Greedily add remaining sets to maximize comparabilities
        all_masks = list(range(N))
        rng.shuffle(all_masks)
        
        while len(family) < m:
            best_add = None
            best_deg = -1
            for cand in all_masks:
                if cand in seen:
                    continue
                deg = 0
                base_c = cand * N
                for s in family:
                    if comp[base_c + s]:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_add = cand
            if best_add is not None:
                family.append(best_add)
                seen.add(best_add)
            else:
                for cand in all_masks:
                    if cand not in seen:
                        family.append(cand)
                        seen.add(cand)
                        break
                if len(family) >= m:
                    break
        
        return family[:m]
    
    def local_search(family, rng, time_limit):
        """Run greedy local search until no improvement or time limit."""
        cur_score = compute_score(family)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(m):
                if time.time() > time_limit:
                    break
                old = family[i]
                deg_old = get_degree(old, family, i)
                
                # Try all candidates (fast with precomputed matrix)
                best_new = None
                best_deg = -1
                for new in range(N):
                    if new == old or new in family:
                        continue
                    base_n = new * N
                    deg = 0
                    for j, s in enumerate(family):
                        if j != i and comp[base_n + s]:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_new = new
                
                if best_new is not None and best_deg > deg_old:
                    family[i] = best_new
                    cur_score += (best_deg - deg_old)
                    improved = True
        
        return cur_score
    
    def anneal_search(family, rng, time_limit):
        """Simulated annealing for escaping local optima."""
        cur_score = compute_score(family)
        temp = 8.0
        cooling = 0.97
        family_set = set(family)
        
        while temp > 0.5 and time.time() < time_limit:
            i = rng.randint(0, m - 1)
            old = family[i]
            deg_old = get_degree(old, family, i)
            
            new = rng.randint(0, N - 1)
            if new == old or new in family_set:
                temp *= cooling
                continue
            
            deg_new = get_degree(new, family, i)
            delta = deg_new - deg_old
            
            if delta > 0 or rng.random() < pow(2.718281828, delta / temp):
                family_set.discard(old)
                family[i] = new
                family_set.add(new)
                cur_score += delta
            
            temp *= cooling
        
        return cur_score
    
    best_family = None
    best_score = -1
    
    # Multi-restart search
    rng = random.Random(42)
    num_restarts = 0
    
    while time.time() < deadline - 5:
        num_restarts += 1
        family = make_chain_family(rng)
        
        # Phase 1: Greedy local search
        remaining = deadline - 5 - time.time()
        phase1_time = time.time() + max(1.0, remaining * 0.4)
        score1 = local_search(family, rng, phase1_time)
        
        # Phase 2: Annealing
        if time.time() < deadline - 3:
            phase2_time = time.time() + max(0.5, (deadline - 3 - time.time()) * 0.5)
            score2 = anneal_search(family, rng, phase2_time)
            score1 = max(score1, score2)
        
        # Phase 3: Final greedy polish
        if time.time() < deadline - 1:
            phase3_time = deadline - 1
            score3 = local_search(family, rng, phase3_time)
            score1 = max(score1, score3)
        
        if score1 > best_score:
            best_score = score1
            best_family = list(family)
        
        # If we've found a very good solution, do more polish
        if best_score >= 350:
            break
    
    # Final verification and polish
    if best_family is not None and time.time() < deadline - 0.5:
        final_score = local_search(best_family, rng, deadline - 0.5)
        if final_score > best_score:
            best_score = final_score
    
    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(n) if (mask >> i) & 1]
        result.append(lst)
    
    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
