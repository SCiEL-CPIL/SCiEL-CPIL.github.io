import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    all_blocks = list(itertools.combinations(all_points, k))
    n_blocks = len(all_blocks)

    # Precompute block -> subset indices
    block_to_subs = []
    sub_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, b in enumerate(all_blocks):
        subs_in_b = []
        for sub in itertools.combinations(b, t):
            idx = t_sub_to_idx[sub]
            subs_in_b.append(idx)
            sub_to_blocks[idx].append(b_idx)
        block_to_subs.append(subs_in_b)
        
    # Convert to sets for faster lookup in local search
    block_to_subs_set = [set(subs) for subs in block_to_subs]

    # Greedy Initialization
    covered = [0] * n_t
    blocks_in_sol = []
    block_in_sol = [False] * n_blocks
    weights = [1.0] * n_t

    while any(c == 0 for c in covered) and time_ok():
        best_block_idx = -1
        best_score = -1.0

        for b_idx in range(n_blocks):
            score = 0.0
            for s_idx in block_to_subs[b_idx]:
                if covered[s_idx] == 0:
                    score += weights[s_idx]
            if score > best_score:
                best_score = score
                best_block_idx = b_idx

        if best_block_idx == -1 or best_score == 0:
            break

        blocks_in_sol.append(best_block_idx)
        block_in_sol[best_block_idx] = True
        
        for s_idx in block_to_subs[best_block_idx]:
            covered[s_idx] += 1
            if covered[s_idx] == 1:
                weights[s_idx] *= 0.7

    # Fallback
    while any(c == 0 for c in covered) and time_ok():
        b_idx = random.randint(0, n_blocks - 1)
        if not block_in_sol[b_idx]:
            blocks_in_sol.append(b_idx)
            block_in_sol[b_idx] = True
            for s_idx in block_to_subs[b_idx]:
                covered[s_idx] += 1

    def is_redundant(b_idx):
        for s_idx in block_to_subs[b_idx]:
            if covered[s_idx] == 1:
                return False
        return True

    def try_remove(b_idx):
        if is_redundant(b_idx):
            for s_idx in block_to_subs[b_idx]:
                covered[s_idx] -= 1
            blocks_in_sol.remove(b_idx)
            block_in_sol[b_idx] = False
            return True
        return False

    def full_deletion_pass():
        changed = True
        while changed and time_ok():
            changed = False
            for b_idx in list(blocks_in_sol):
                if try_remove(b_idx):
                    changed = True
                    break

    full_deletion_pass()
    
    best_sol_size = len(blocks_in_sol)
    best_sol = blocks_in_sol.copy()
    
    # Local Search
    while time_ok():
        if not blocks_in_sol:
            break
        random.shuffle(blocks_in_sol)
        found_improvement = False
        
        for i, b_idx in enumerate(blocks_in_sol):
            if not time_ok():
                break
            
            excl_subs = []
            for s_idx in block_to_subs[b_idx]:
                if covered[s_idx] == 1:
                    excl_subs.append(s_idx)
            
            if not excl_subs:
                try_remove(b_idx)
                found_improvement = True
                break
            
            candidates = sub_to_blocks[excl_subs[0]]
            valid_candidates = []
            for c_idx in candidates:
                if block_in_sol[c_idx]:
                    continue
                # Check if c_idx covers all excl_subs
                valid = True
                for s_idx in excl_subs[1:]:
                    if s_idx not in block_to_subs_set[c_idx]:
                        valid = False
                        break
                if valid:
                    valid_candidates.append(c_idx)
            
            if not valid_candidates:
                continue
                
            random.shuffle(valid_candidates)
            for c_idx in valid_candidates[:5]:
                old_covered = covered[:]
                old_blocks = blocks_in_sol[:]
                old_block_in_sol = block_in_sol[:]
                
                for s_idx in block_to_subs[b_idx]:
                    covered[s_idx] -= 1
                blocks_in_sol.remove(b_idx)
                block_in_sol[b_idx] = False
                
                for s_idx in block_to_subs[c_idx]:
                    covered[s_idx] += 1
                blocks_in_sol.append(c_idx)
                block_in_sol[c_idx] = True
                
                full_deletion_pass()
                
                if len(blocks_in_sol) < best_sol_size:
                    best_sol_size = len(blocks_in_sol)
                    best_sol = blocks_in_sol.copy()
                    found_improvement = True
                    break
                else:
                    covered = old_covered
                    blocks_in_sol = old_blocks
                    block_in_sol = old_block_in_sol
            
            if found_improvement:
                break

    return [list(all_blocks[b_idx]) for b_idx in best_sol]
