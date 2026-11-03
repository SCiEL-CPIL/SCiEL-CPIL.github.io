import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_subs = list(itertools.combinations(range(v), t))
    num_subs = len(all_subs)
    
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    block_to_subs_list = []
    block_to_subs_set = []
    for b in all_blocks:
        subs = [sub_to_idx[s] for s in itertools.combinations(b, t)]
        block_to_subs_list.append(subs)
        block_to_subs_set.append(set(subs))
        
    sub_to_blocks = [[] for _ in range(num_subs)]
    for i, subs in enumerate(block_to_subs_list):
        for s in subs:
            sub_to_blocks[s].append(i)

    def run_search():
        coverage = [0] * num_subs
        current_blocks = []
        current_blocks_set = set()
        
        # Greedy Phase
        while True:
            if min(coverage) > 0:
                break
                
            best_block = -1
            max_gain = -1
            
            for i in range(num_blocks):
                gain = 0
                for s in block_to_subs_list[i]:
                    if coverage[s] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = i
                    if gain == len(block_to_subs_list[i]):
                        break
            
            if best_block == -1:
                break
                
            current_blocks.append(best_block)
            current_blocks_set.add(best_block)
            for s in block_to_subs_list[best_block]:
                coverage[s] += 1
            
            if time.time() > deadline:
                return current_blocks

        # Local Search Phase
        improved = True
        while improved:
            improved = False
            
            # Removal
            for idx in range(len(current_blocks) - 1, -1, -1):
                b = current_blocks[idx]
                can_remove = True
                for s in block_to_subs_list[b]:
                    if coverage[s] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for s in block_to_subs_list[b]:
                        coverage[s] -= 1
                    current_blocks.pop(idx)
                    current_blocks_set.remove(b)
                    improved = True
            
            # Replacement
            for idx in range(len(current_blocks)):
                b = current_blocks[idx]
                unique_subs = []
                for s in block_to_subs_list[b]:
                    if coverage[s] == 1:
                        unique_subs.append(s)
                
                if not unique_subs:
                    continue
                
                s_key = unique_subs[0]
                candidates = sub_to_blocks[s_key]
                
                for b_new in candidates:
                    if b_new == b:
                        continue
                    if b_new in current_blocks_set:
                        continue
                    
                    if all(s in block_to_subs_set[b_new] for s in unique_subs):
                        # Replace
                        for s in block_to_subs_list[b]:
                            coverage[s] -= 1
                        for s in block_to_subs_list[b_new]:
                            coverage[s] += 1
                        current_blocks[idx] = b_new
                        current_blocks_set.remove(b)
                        current_blocks_set.add(b_new)
                        improved = True
                        break
                        
                if time.time() > deadline:
                    return current_blocks

        return current_blocks

    blocks = run_search()
    if not blocks:
        return []
        
    return [list(all_blocks[b]) for b in blocks]
