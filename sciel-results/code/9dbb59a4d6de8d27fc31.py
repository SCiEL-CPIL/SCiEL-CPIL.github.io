import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    # 1. Precompute all t-subsets (5-subsets of 19 points)
    all_t = list(itertools.combinations(range(v), t))
    N = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # 2. Generate all base blocks (orbits) under cyclic group Z_v
    # Since v=19 is prime, all orbits have size v unless the block is invariant under a rotation,
    # which is impossible for k < v. So each orbit has exactly v blocks.
    
    # We generate canonical representatives for each orbit.
    # A subset is canonical if its rotation list is lexicographically smallest.
    
    def get_rotations(subset):
        """Generate all v rotations of a subset."""
        s = sorted(subset)
        rots = []
        for i in range(v):
            rot = tuple((x + i) % v for x in s)
            rots.append(rot)
        return rots

    # To efficiently find canonical reps, we can iterate through all k-subsets
    # and check if it's the min of its orbit.
    # However, generating all k-subsets is 75582. This is fast enough.
    
    canonical_reps = set()
    orbits = [] # List of (canonical_rep_tuple, list_of_blocks_in_orbit)
    
    # Optimization: Instead of checking all subsets, we can just generate them 
    # and group them. But 75k is small. Let's do it simply.
    
    # To speed up canonical check, we can use a trick:
    # For a subset S, if S is the lexicographically smallest in its orbit, it's canonical.
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    
    # Map from tuple to orbit index
    block_to_orbit = {}
    orbit_blocks = []
    
    # We need to identify orbits.
    # We'll iterate through all k-subsets. If we haven't seen this orbit, add it.
    # How to know if we've seen it? Use the canonical representative.
    
    # Precompute rotation offsets
    # For each subset, compute its canonical form.
    
    # This might be slow in pure Python if done naively for 75k items.
    # Let's optimize: 
    # A subset is canonical if for all i in 1..v-1, sorted(S) < sorted((S+i)%v)
    
    # Actually, there is a faster way:
    # Just generate all subsets.
    # For each subset, compute its canonical representative.
    # If it's new, add to list.
    
    # Let's precompute the "cover" for each orbit.
    # An orbit covers a set of t-subsets.
    # We want to select a set of orbits such that the union of their covered t-subsets is All.
    
    # First, build the list of orbits and their coverage.
    
    orbit_cover = [] # List of sets of t-subset indices
    orbit_canonicals = [] # List of canonical tuples
    
    seen_canonicals = set()
    
    # To make this faster, we can use the fact that we only need to check 
    # if a subset is the min of its orbit.
    
    # Let's just iterate. 75,582 iterations is fine.
    
    for s in all_k_subsets:
        # Check if s is canonical
        # s is a tuple of sorted ints.
        # Generate rotations
        is_canonical = True
        min_rot = s
        for i in range(1, v):
            rot = tuple(sorted((x + i) % v for x in s))
            if rot < min_rot:
                is_canonical = False
                break
            if rot > min_rot:
                # We can't break yet, need to check all? 
                # Actually if rot > min_rot, min_rot is still min so far.
                pass
        
        if is_canonical:
            # This is a canonical representative of a new orbit
            seen_canonicals.add(s)
            
            # Compute coverage of this orbit
            # The orbit consists of all rotations of s.
            # We need the union of t-subsets covered by all blocks in this orbit.
            
            cov_set = set()
            for i in range(v):
                block = tuple((x + i) % v for x in s)
                # Find all t-subsets in this block
                for ts in itertools.combinations(block, t):
                    if ts in t_idx:
                        cov_set.add(t_idx[ts])
            
            orbit_cover.append(np.array(list(cov_set), dtype=np.int32))
            orbit_canonicals.append(s)
            
            # Map all blocks in this orbit to this orbit index
            for i in range(v):
                block = tuple((x + i) % v for x in s)
                block_to_orbit[block] = len(orbit_cover) - 1

    num_orbits = len(orbit_cover)
    
    # 3. Greedy Set Cover on Orbits
    # We want to select a subset of orbits that covers all t-subsets.
    # This is a smaller set cover problem: num_orbits (~4000) vs num_blocks (~75000).
    
    # Precompute for each t-subset, which orbits cover it.
    # tsub_to_orbits = [[] for _ in range(N)]
    # This is expensive to build directly.
    # Instead, we can do greedy by iterating orbits and picking the one with max gain.
    # But we need to check gain efficiently.
    
    # Let's use a boolean array for uncovered t-subsets.
    uncovered = np.ones(N, dtype=bool)
    selected_orbits = []
    
    # To speed up greedy, we can precompute the size of each orbit's coverage.
    # And maybe use a priority queue? 
    # But the gain changes as we select orbits.
    
    # Simple greedy:
    # At each step, pick the orbit that covers the most currently uncovered t-subsets.
    
    # Optimization: 
    # Precompute orbit_cover as boolean masks? No, N is 11628. 
    # orbit_cover is a list of arrays of indices.
    
    # We can represent coverage as a bitset? 
    # N=11628 bits is ~1.5KB per orbit. 4000 orbits * 1.5KB = 6MB. 
    # This is very manageable.
    
    # Let's convert orbit_cover to a list of bitmasks (using Python ints or numpy arrays)
    # Using numpy bool arrays for intersection is fast.
    
    # Actually, just using the list of indices and checking `uncovered[cov]` is fine.
    
    # Let's run the greedy loop.
    # To make it faster, we can stop if we run out of time.
    
    while np.any(uncovered):
        if time.time() > deadline - 20:
            break
            
        best_orbit = -1
        best_gain = 0
        
        # Iterate through all orbits
        for oi in range(num_orbits):
            if oi in selected_orbits: # This check is slow if selected_orbits is large
                continue
            # Better: keep a list of available orbits
            pass
        
        # Let's maintain a list of available orbits
        # Actually, just iterating all and checking a set is okay for 4000 items.
        selected_set = set(selected_orbits)
        
        for oi in range(num_orbits):
            if oi in selected_set:
                continue
            cov = orbit_cover[oi]
            gain = np.sum(uncovered[cov])
            if gain > best_gain:
                best_gain = gain
                best_orbit = oi
        
        if best_orbit == -1 or best_gain == 0:
            break
            
        selected_orbits.append(best_orbit)
        uncovered[orbit_cover[best_orbit]] = False

    # 4. Expand selected orbits to full blocks
    final_blocks = []
    for oi in selected_orbits:
        s = orbit_canonicals[oi]
        for i in range(v):
            block = tuple((x + i) % v for x in s)
            final_blocks.append(list(block))
    
    # 5. Local Search to Prune
    # Now we have a valid covering. We want to remove redundant blocks.
    # We can use the same local search logic as before, but starting from this better solution.
    
    # Build incidence for the *selected* blocks only? 
    # No, we need to know if removing a block breaks coverage.
    # We need to know for each t-subset, how many of the *current* blocks cover it.
    
    # Let's build a cover_count for the current solution.
    cover_count = np.zeros(N, dtype=np.int32)
    block_indices = [] # List of block tuples
    
    # Map block tuple to index in final_blocks
    block_to_idx = {}
    
    for i, b in enumerate(final_blocks):
        bt = tuple(b)
        block_to_idx[bt] = i
        for ts in itertools.combinations(bt, t):
            cover_count[t_idx[ts]] += 1
            
    # Verify validity
    if np.any(cover_count == 0):
        # Fallback: if greedy on orbits failed to cover everything (shouldn't happen),
        # add blocks greedily to fix.
        uncovered_idx = np.where(cover_count == 0)[0]
        for ti in uncovered_idx:
            # Find a block covering this t-subset
            # We can search all_blocks
            ts = all_t[ti]
            for b in all_k_subsets:
                if set(ts).issubset(set(b)):
                    bt = tuple(b)
                    if bt not in block_to_idx:
                        idx = len(final_blocks)
                        final_blocks.append(list(b))
                        block_to_idx[bt] = idx
                        cover_count[t_idx[ts]] += 1
                        # Update other counts for this block
                        for ts2 in itertools.combinations(b, t):
                            cover_count[t_idx[ts2]] += 1
                    break
            if time.time() > deadline - 5:
                break

    # Local Search: Try to remove blocks
    # We iterate through blocks, and if removing one doesn't break coverage, remove it.
    # Repeat until no more removals possible.
    # Then try swaps? Swaps are harder to implement correctly in this structure.
    # Let's stick to removal first, then maybe a simple swap heuristic.
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 10:
            break
            
        for i in range(len(final_blocks) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
                
            b = final_blocks[i]
            bt = tuple(b)
            
            # Check if removal breaks coverage
            can_remove = True
            for ts in itertools.combinations(b, t):
                if cover_count[t_idx[ts]] == 1:
                    can_remove = False
                    break
                    
            if can_remove:
                # Remove it
                for ts in itertools.combinations(b, t):
                    cover_count[t_idx[ts]] -= 1
                final_blocks.pop(i)
                del block_to_idx[bt]
                improved = True

    return final_blocks

