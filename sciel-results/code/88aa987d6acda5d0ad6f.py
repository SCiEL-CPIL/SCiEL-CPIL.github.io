"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random

def construct():
    _start = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

    def _remaining():
        return _DEADLINE - time.time()

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def _build_triples(vals):
        """Return list of (i,j,k) indices where vals[i]+vals[j]==vals[k], i<=j allowed."""
        vset = set(vals)
        triples = []
        n = len(vals)
        # For each pair (i,j) with i<=j, check if vals[i]+vals[j] is in vals
        # But we want to find all triples efficiently
        # Use a dict: value -> index
        vidx = {v: i for i, v in enumerate(vals)}
        for i in range(n):
            for j in range(i, n):
                s = vals[i] + vals[j]
                if s in vidx:
                    k = vidx[s]
                    # a+b=c, so c = s. The triple is (i,j,k) where vals[i]+vals[j]=vals[k]
                    triples.append((i, j, k))
        return triples

    def _is_sum_free(selected, vals):
        """Check if selected (set of indices) is sum-free."""
        if len(selected) < 3:
            return True
        vset = set(selected)
        for a in vset:
            for b in vset:
                if a + b in vset:
                    return False
        return True

    def _greedy_remove(vals, rng):
        """Greedy: start with all, iteratively remove element in most violating triples."""
        n = len(vals)
        present = list(range(n))  # all indices present
        present_set = set(present)
        
        # Build all triples
        vidx = {v: i for i, v in enumerate(vals)}
        all_triples = []
        for i in range(n):
            for j in range(i, n):
                s = vals[i] + vals[j]
                if s in vidx:
                    k = vidx[s]
                    all_triples.append((i, j, k))
        
        # Iteratively remove
        while True:
            # Count how many active triples each element is in
            if not all_triples:
                break
            counts = {}
            active_triples = []
            for (i, j, k) in all_triples:
                if i in present_set and j in present_set and k in present_set:
                    active_triples.append((i, j, k))
                    counts[i] = counts.get(i, 0) + 1
                    counts[j] = counts.get(j, 0) + 1
                    counts[k] = counts.get(k, 0) + 1
            if not active_triples:
                break
            # Remove element with highest count (tie-break randomly)
            max_count = max(counts.values())
            candidates = [e for e, c in counts.items() if c == max_count]
            to_remove = rng.choice(candidates)
            present_set.discard(to_remove)
            present.remove(to_remove)
        
        return present_set

    def _try_add_back(removed_set, present_set, vals):
        """Try to add back elements from removed_set if they don't create conflicts."""
        n = len(vals)
        vidx = {v: i for i, v in enumerate(vals)}
        improved = True
        while improved:
            improved = False
            for r in list(removed_set):
                # Check if adding r creates any conflict with current present_set
                conflict = False
                for a in present_set:
                    # Check a + r = c for c in present_set
                    if (a + r) in vidx and vidx[a + r] in present_set:
                        conflict = True
                        break
                    # Check r + a = c (same as above)
                    # Check r + r = c
                if not conflict:
                    # Also need to check r+r in present_set
                    if (2 * r) in vidx and vidx[2 * r] in present_set:
                        conflict = True
                    # Check a + b = r for a,b in present_set
                    if not conflict:
                        for a in present_set:
                            if (r - a) in vidx and vidx[r - a] in present_set:
                                conflict = True
                                break
                if not conflict:
                    present_set.add(r)
                    removed_set.discard(r)
                    improved = True
        return present_set, removed_set

    def _local_search_swap(present_set, removed_set, vals, rng):
        """Try 1-for-2 and 2-for-3 swaps to improve size."""
        n = len(vals)
        vidx = {v: i for i, v in enumerate(vals)}
        improved = True
        while improved and _remaining() > 5.0:
            improved = False
            # Try: remove one from present, add two from removed
            for p in list(present_set):
                for r1 in list(removed_set):
                    for r2 in removed_set:
                        if r1 >= r2:
                            continue
                        # Check if (present - {p} + {r1, r2}) is sum-free
                        new_set = present_set - {p} | {r1, r2}
                        if _is_sum_free(new_set, vals):
                            present_set = new_set
                            removed_set = removed_set - {r1, r2} | {p}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Try: remove two from present, add three from removed
            plist = list(present_set)
            for i in range(len(plist)):
                for j in range(i + 1, len(plist)):
                    p1, p2 = plist[i], plist[j]
                    for r1 in removed_set:
                        for r2 in removed_set:
                            if r1 >= r2:
                                continue
                            for r3 in removed_set:
                                if r3 <= r2:
                                    continue
                                new_set = present_set - {p1, p2} | {r1, r2, r3}
                                if _is_sum_free(new_set, vals):
                                    present_set = new_set
                                    removed_set = removed_set - {r1, r2, r3} | {p1, p2}
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return present_set, removed_set

    def _solve_instance(vals, rng):
        """Solve one instance with multiple strategies and restarts."""
        n = len(vals)
        best = set()
        
        # Strategy 1: Greedy remove with multiple random orderings
        for attempt in range(20):
            if _remaining() < 10.0:
                break
            ps = _greedy_remove(vals, rng)
            rs = set(range(n)) - ps
            ps, rs = _try_add_back(rs, ps, vals)
            ps, rs = _local_search_swap(ps, rs, vals, rng)
            if len(ps) > len(best):
                best = set(ps)
        
        # Strategy 2: Start with odd/even split and improve
        odds = {i for i in range(n) if vals[i] % 2 != 0}
        evens = {i for i in range(n) if vals[i] % 2 == 0}
        for start_set in [odds, evens]:
            if _remaining() < 10.0:
                break
            ps = set(start_set)
            rs = set(range(n)) - ps
            ps, rs = _try_add_back(rs, ps, vals)
            ps, rs = _local_search_swap(ps, rs, vals, rng)
            if len(ps) > len(best):
                best = set(ps)
        
        # Strategy 3: Random subset of size ~n/2, then improve
        for attempt in range(10):
            if _remaining() < 10.0:
                break
            ps = set(rng.sample(range(n), n // 2))
            rs = set(range(n)) - ps
            # First make it sum-free by removing conflicts
            while True:
                conflict_found = False
                vset = set(ps)
                for a in list(vset):
                    for b in list(vset):
                        s = vals[a] + vals[b]
                        if s in {vals[k] for k in vset}:
                            # Remove a
                            ps.discard(a)
                            conflict_found = True
                            break
                    if conflict_found:
                        break
                if not conflict_found:
                    break
            rs = set(range(n)) - ps
            ps, rs = _try_add_back(rs, ps, vals)
            ps, rs = _local_search_swap(ps, rs, vals, rng)
            if len(ps) > len(best):
                best = set(ps)
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if _remaining() < 15.0:
            # Fallback: just odds
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue
        vals = instance(size, seed)
        rng = _random.Random(seed + 12345)
        best = _solve_instance(vals, rng)
        results.append([vals[i] for i in sorted(best)])
    
    return results
# EVOLVE-BLOCK-END
