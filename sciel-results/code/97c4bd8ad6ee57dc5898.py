import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def is_valid_superperm(word, n):
        if len(word) < n:
            return False
        perms = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n:
                perms.add(window)
        return len(perms) == n * (n-1) * (n-2) * (n-3) * (n-4) * (n-5) * (n-6) if n==7 else len(perms) == __import__('math').factorial(n)

    # Standard construction function
    def standard_construct(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    # GNN construction
    def gnn_construct(n):
        word = [1]
        for m in range(2, n + 1):
            if time.time() > deadline:
                return standard_construct(n) # Fallback
                
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
                    
            blocks = []
            for p in perms:
                block = list(p) + [m] + list(p)
                blocks.append(block)
                
            if len(blocks) <= 1:
                ordered_blocks = blocks
            else:
                # GNN
                # Try multiple starts to find a better path
                best_len = float('inf')
                best_ordered = None
                
                # For small m, try all starts. For large m, try a few.
                starts = range(len(blocks)) if len(blocks) < 50 else range(0, len(blocks), 10)
                
                for start_idx in starts:
                    remaining = list(range(len(blocks)))
                    remaining.remove(start_idx)
                    current_idx = start_idx
                    ordered_indices = [current_idx]
                    
                    while remaining:
                        curr_block = blocks[current_idx]
                        best_next_idx = -1
                        best_overlap = -1
                        
                        for next_idx in remaining:
                            next_block = blocks[next_idx]
                            max_k = min(len(curr_block), len(next_block))
                            found_overlap = 0
                            for k in range(max_k, 0, -1):
                                if curr_block[-k:] == next_block[:k]:
                                    found_overlap = k
                                    break
                            
                            if found_overlap > best_overlap:
                                best_overlap = found_overlap
                                best_next_idx = next_idx
                                
                        if best_next_idx != -1:
                            ordered_indices.append(best_next_idx)
                            remaining.remove(best_next_idx)
                            current_idx = best_next_idx
                        else:
                            break
                            
                    # Calculate total length
                    total_len = 0
                    for i, idx in enumerate(ordered_indices):
                        if i == 0:
                            total_len += len(blocks[idx])
                        else:
                            prev_idx = ordered_indices[i-1]
                            prev_block = blocks[prev_idx]
                            curr_block = blocks[idx]
                            max_k = min(len(prev_block), len(curr_block))
                            overlap = 0
                            for k in range(max_k, 0, -1):
                                if prev_block[-k:] == curr_block[:k]:
                                    overlap = k
                                    break
                            total_len += len(curr_block) - overlap
                            
                    if total_len < best_len:
                        best_len = total_len
                        best_ordered = [blocks[i] for i in ordered_indices]
                        
                ordered_blocks = best_ordered
                
            new_word = []
            for i, block in enumerate(ordered_blocks):
                if i == 0:
                    new_word.extend(block)
                else:
                    prev_block = ordered_blocks[i-1]
                    max_k = min(len(prev_block), len(block))
                    overlap = 0
                    for k in range(max_k, 0, -1):
                        if prev_block[-k:] == block[:k]:
                            overlap = k
                            break
                    new_word.extend(block[overlap:])
                    
            word = new_word
            
        return word

    # Try GNN
    try:
        word = gnn_construct(n)
        if is_valid_superperm(word, n):
            return "".join(map(str, word))
    except Exception:
        pass

    # Fallback to standard
    word = standard_construct(n)
    return "".join(map(str, word))
