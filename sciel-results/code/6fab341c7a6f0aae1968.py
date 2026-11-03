import heapq
import itertools
import time

def construct(q, n, radius):
    total_words = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    def get_neighbors(word, radius):
        # Yield all words within Hamming distance radius
        yield word
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val == word[i]:
                        continue
                    new_word = list(word)
                    new_word[i] = val
                    yield tuple(new_word)
        # If radius > 1, we would need more, but task is radius 1.
        
    covered_by = [[] for _ in range(total_words)]
    coverers = [[] for _ in range(total_words)]
    
    for c_idx, c in enumerate(words):
        for w in get_neighbors(c, radius):
            w_idx = word_to_idx[w]
            covered_by[c_idx].append(w_idx)
            coverers[w_idx].append(c_idx)
            
    # Greedy Set Cover
    cur_count = [0] * total_words
    for c in range(total_words):
        cur_count[c] = len(covered_by[c])
        
    heap = [(-cur_count[c], c) for c in range(total_words)]
    heapq.heapify(heap)
    
    selected = []
    uncovered = set(range(total_words))
    uncovered_count = total_words
    
    start_time = time.time()
    deadline = start_time + 100.0 # 20s margin
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        neg_cnt, c = heapq.heappop(heap)
        cnt = -neg_cnt
        
        if cnt != cur_count[c]:
            continue
            
        selected.append(c)
        
        for w in covered_by[c]:
            if w in uncovered:
                uncovered.remove(w)
                uncovered_count -= 1
                for c_prime in coverers[w]:
                    if c_prime != c:
                        cur_count[c_prime] -= 1
                        heapq.heappush(heap, (-cur_count[c_prime], c_prime))
                        
    # Pruning Phase
    cover_count = [0] * total_words
    selected_set = set(selected)
    for c in selected:
        for w in covered_by[c]:
            cover_count[w] += 1
            
    changed = True
    prune_deadline = time.time() + 10.0
    while changed:
        if time.time() > prune_deadline:
            break
        changed = False
        # Iterate over a list of selected to allow safe removal
        for c in list(selected_set):
            can_remove = True
            for w in covered_by[c]:
                if cover_count[w] == 1:
                    can_remove = False
                    break
                    
            if can_remove:
                selected_set.remove(c)
                selected.remove(c)
                for w in covered_by[c]:
                    cover_count[w] -= 1
                changed = True
                
    return [list(words[c]) for c in selected]
