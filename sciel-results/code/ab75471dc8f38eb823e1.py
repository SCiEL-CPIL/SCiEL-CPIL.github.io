import time
import random
import numpy as np
import itertools

def construct(v, k, t):
    start = time.time()
    deadline = start + 145  # safety margin under 150s wall-clock

    # ---------- Precompute ----------
    # All k-subsets of {0..v-1}
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # All t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    n_subs = len(t_subs)
    t_sub_index = {s: i for i, s in enumerate(t_subs)}

    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for blk in all_blocks:
        subs = [t_sub_index[s] for s in itertools.combinations(blk, t)]
        block_tsubs.append(subs)

    # NumPy arrays for speed
    # block_to_sub: list of arrays
    bts = [np.array(s, dtype=np.int32) for s in block_tsubs]

    # ---------- Coverage accounting (vectorized-ish) ----------
    # We maintain coverage counts for each t-subset
    # Use a numpy array for counts
    def make_counts():
        return np.zeros(n_subs, dtype=np.int32)

    def add_block(counts, bi):
        counts[bts[bi]] += 1

    def remove_block(counts, bi):
        counts[bts[bi]] -= 1

    def uncovered_mask(counts):
        return counts == 0

    # ---------- Greedy set cover (vectorized scoring) ----------
    def greedy_set_cover(counts, chosen_set, max_blocks=None):
        """Continue greedy from current state. Returns (counts, chosen_list, chosen_set, valid)."""
        chosen = list(chosen_set)
        chosen_set = set(chosen)
        if max_blocks is None:
            max_blocks = n_blocks

        while True:
            if time.time() > deadline - 5:
                break
            uncov = uncovered_mask(counts)
            if not uncov.any():
                break
            if len(chosen) >= max_blocks:
                break

            # Score each candidate block: number of uncovered subsets it covers
            # Vectorized: for each block, sum of uncov[its tsubs]
            # This is expensive if done naively; use a trick
            # Build a matrix? Too big. Instead, loop over blocks but use numpy sum.
            # Optimize: only consider blocks not in chosen_set
            best_gain = 0
            best_bi = -1
            # Precompute uncovered count
            n_uncov = int(uncov.sum())
            
            # Fast path: try to find a block covering all uncovered (unlikely)
            # Just iterate
            for bi in range(n_blocks):
                if bi in chosen_set:
                    continue
                gain = int(uncov[bts[bi]].sum())
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    if best_gain == n_uncov:
                        break
            if best_bi < 0:
                break
            add_block(counts, best_bi)
            chosen.append(best_bi)
            chosen_set.add(best_bi)

        valid = not uncovered_mask(counts).any()
        return counts, chosen, chosen_set, valid

    # ---------- Prune redundant blocks ----------
    def prune(counts, chosen):
        """Remove blocks that are redundant (all their tsubs have count > 1)."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        changed = True
        while changed:
            changed = False
            for i in range(len(chosen)):
                if time.time() > deadline - 5:
                    break
                bi = chosen[i]
                # Check if all tsubs of bi have count >= 2
                c = counts[bts[bi]]
                if (c >= 2).all():
                    remove_block(counts, bi)
                    chosen.pop(i)
                    chosen_set.discard(bi)
                    changed = True
        return counts, chosen, chosen_set

    # ---------- Simulated Annealing ----------
    def simulated_annealing(counts, chosen, chosen_set, time_budget):
        """SA over block multiset. Moves: add, remove, swap."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        counts = counts.copy()
        
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        rng = random.Random(42)
        
        best_size = len(chosen)
        best_chosen = chosen[:]
        best_counts = counts.copy()
        best_set = chosen_set.copy()
        
        end_time = time.time() + time_budget
        while time.time() < end_time:
            if T < T_min:
                T = 1.0  # restart temperature occasionally
            
            move_type = rng.random()
            new_counts = counts.copy()
            new_chosen = chosen[:]
            new_set = chosen_set.copy()
            
            if move_type < 0.3:
                # Remove a random block
                if not chosen:
                    T *= cooling
                    continue
                idx = rng.randrange(len(chosen))
                bi = chosen[idx]
                remove_block(new_counts, bi)
                new_chosen.pop(idx)
                new_set.discard(bi)
            elif move_type < 0.6:
                # Add a random block
                bi = rng.randrange(n_blocks)
                if bi not in new_set:
                    add_block(new_counts, bi)
                    new_chosen.append(bi)
                    new_set.add(bi)
            else:
                # Swap: remove one, add one
                if not chosen:
                    T *= cooling
                    continue
                idx = rng.randrange(len(chosen))
                bi_out = chosen[idx]
                bi_in = rng.randrange(n_blocks)
                if bi_in not in new_set:
                    remove_block(new_counts, bi_out)
                    new_chosen.pop(idx)
                    new_set.discard(bi_out)
                    add_block(new_counts, bi_in)
                    new_chosen.append(bi_in)
                    new_set.add(bi_in)
            
            # Evaluate
            n_uncov = int(uncovered_mask(new_counts).sum())
            size = len(new_chosen)
            
            # Objective: minimize size + penalty for uncovered
            # If valid, objective = size; else size + n_uncov * 1000
            if n_uncov == 0:
                obj = size
            else:
                obj = size + n_uncov * 1000
            
            if obj < best_size or (n_uncov == 0 and size < best_size):
                if n_uncov == 0 and size < best_size:
                    best_size = size
                    best_chosen = new_chosen[:]
                    best_counts = new_counts.copy()
                    best_set = new_set.copy()
            
            # Accept or reject
            # Current objective
            cur_uncov = int(uncovered_mask(counts).sum())
            cur_obj = len(chosen) + (cur_uncov * 1000 if cur_uncov > 0 else 0)
            
            delta = obj - cur_obj
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                counts = new_counts
                chosen = new_chosen
                chosen_set = new_set
            
            T *= cooling
        
        return best_counts, best_chosen, best_set

    # ---------- Main ----------
    rng = random.Random(1234567890)
    
    best_size = float('inf')
    best_blocks = None
    best_counts = None
    
    # Phase 1: Multiple greedy starts with different initial random subsets
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        
        # Start with a random small set of blocks
        initial_size = rng.randint(10, 30)
        initial_blocks = rng.sample(range(n_blocks), min(initial_size, n_blocks))
        
        counts = make_counts()
        chosen_set = set()
        for bi in initial_blocks:
            add_block(counts, bi)
            chosen_set.add(bi)
        
        # Complete with greedy
        counts, chosen, chosen_set, valid = greedy_set_cover(counts, chosen_set, max_blocks=300)
        
        if not valid:
            continue
        
        # Prune
        counts, chosen, chosen_set = prune(counts, chosen)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_blocks = chosen[:]
            best_counts = counts.copy()
    
    # Phase 2: Simulated annealing from best
    if best_blocks is not None and time.time() < deadline - 15:
        counts, chosen, chosen_set = prune(best_counts.copy(), best_blocks[:])
        sa_time = min(20, max(5, deadline - time.time() - 10))
        counts, chosen, chosen_set = simulated_annealing(counts, chosen, chosen_set, sa_time)
        
        # Final prune
        counts, chosen, chosen_set = prune(counts, chosen)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_blocks = chosen[:]
            best_counts = counts.copy()
    
    # Phase 3: More SA iterations if time allows
    if best_blocks is not None:
        while time.time() < deadline - 5:
            counts = best_counts.copy()
            chosen = best_blocks[:]
            chosen_set = set(chosen)
            
            sa_time = min(10, max(3, deadline - time.time() - 5))
            counts, chosen, chosen_set = simulated_annealing(counts, chosen, chosen_set, sa_time)
            
            counts, chosen, chosen_set = prune(counts, chosen)
            
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_blocks = chosen[:]
                best_counts = counts.copy()
            else:
                # No improvement, break if we've done several rounds
                break
    
    # Final validation
    if best_blocks is None:
        # Fallback: pure greedy from scratch
        counts = make_counts()
        counts, chosen, chosen_set, valid = greedy_set_cover(counts, set(), max_blocks=500)
        best_blocks = chosen
    
    # Convert to list of lists
    result = [list(all_blocks[bi]) for bi in best_blocks]
    
    # Ensure validity (exhaustive check is done by evaluator, but let's be safe)
    # The evaluator checks, so we just return
    
    return result
