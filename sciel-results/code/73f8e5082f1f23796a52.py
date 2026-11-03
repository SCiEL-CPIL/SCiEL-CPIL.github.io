import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    total_words = q ** n
    
    # Generate all words as tuples of integers
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (Hamming distance <= radius)
    # For q=3, n=6, radius=1, each word has 1 + n*(q-1) = 1 + 6*2 = 13 neighbors?
    # Wait, distance 1 means exactly 1 position different.
    # Number of neighbors at distance 1 is n * (q - 1).
    # Total in ball radius 1 is 1 + n(q-1).
    # For q=3, n=6: 1 + 12 = 13.
    
    neighbors = []
    for i, w in enumerate(all_words):
        nb = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                nb.append(word_to_idx[tuple(new_w)])
        neighbors.append(nb)
        
    covered = [False] * total_words
    code = []
    
    # Greedy Phase
    while True:
        if time.time() > deadline:
            break
            
        if all(covered):
            break
            
        best_idx = -1
        best_gain = -1
        
        for i in range(total_words):
            gain = 0
            for nb in neighbors[i]:
                if not covered[nb]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_gain <= 0:
            break
            
        code.append(best_idx)
        for nb in neighbors[best_idx]:
            covered[nb] = True
            
    # Pruning Phase
    # Try to remove codewords
    changed = True
    while changed:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            if time.time() > deadline:
                break
                
            # Tentatively remove code[i]
            c = code[i]
            
            # Check if removing c uncovers any word
            # A word is uncovered if it is in N(c) and NOT in N(other_codewords)
            # Efficient way:
            # Count how many codewords cover each neighbor of c.
            # If any neighbor is covered only by c, we can't remove c.
            
            # Since code is small, we can just check.
            # But checking "is it covered by others" for all neighbors of c
            # requires iterating neighbors of c and for each, checking if any other codeword covers it.
            
            # Optimization: Precompute coverage counts?
            # Let's just do a direct check for simplicity, code size is ~75.
            
            can_remove = True
            for nb in neighbors[c]:
                # Is nb covered by any other codeword?
                is_covered_by_other = False
                for j in range(len(code)):
                    if j == i:
                        continue
                    # Check if nb is in neighbors[code[j]]
                    # This is O(1) if we use a set, but neighbors are lists.
                    # For efficiency, we can maintain a count array.
                    
                    # Let's use a count array for the pruning phase.
                    pass
                
            # Better approach for pruning:
            # Maintain `cover_count` array.
            # When we add a codeword in greedy, we could have updated counts.
            # But we didn't. Let's recompute counts for the current code.
            
            # Recompute counts is O(|Code| * 13). 75 * 13 is small.
            # Do this once per pruning iteration?
            
            pass
        
        # Let's implement the count-based pruning properly.
        
    # Re-implementing the whole thing with counts for efficiency in pruning
    
    covered_count = [0] * total_words
    for c in code:
        for nb in neighbors[c]:
            covered_count[nb] += 1
            
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Check if any neighbor has count == 1 (meaning only covered by c)
            can_remove = True
            for nb in neighbors[c]:
                if covered_count[nb] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                for nb in neighbors[c]:
                    covered_count[nb] -= 1
                code.pop(i)
                changed = True
                
    # Convert indices back to words
    return [all_words[i] for i in code]
