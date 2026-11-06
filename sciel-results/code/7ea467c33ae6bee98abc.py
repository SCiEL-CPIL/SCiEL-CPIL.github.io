import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    points = list(range(n))
    
    # 1. Generate all triples
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # 2. Generate boxes
    # Precompute partitions of 6 elements into 3 pairs
    def get_partitions_6():
        partitions = []
        def rec(remaining, current):
            if not remaining:
                partitions.append(tuple(sorted(current)))
                return
            first = remaining[0]
            rest = remaining[1:]
            for i, second in enumerate(rest):
                new_pair = (first, second)
                new_remaining = rest[:i] + rest[i+1:]
                rec(new_remaining, current + [new_pair])
        rec(list(range(6)), [])
        return partitions

    partitions_6 = get_partitions_6()
    
    boxes = []
    triple_to_boxes = [[] for _ in range(num_triples)]
    
    for subset in itertools.combinations(points, 6):
        for part in partitions_6:
            p1 = (subset[part[0][0]], subset[part[0][1]])
            p2 = (subset[part[1][0]], subset[part[1][1]])
            p3 = (subset[part[2][0]], subset[part[2][1]])
            
            transversals = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        t = tuple(sorted((a, b, c)))
                        idx = triple_to_idx[t]
                        transversals.append(idx)
            
            box_id = len(boxes)
            boxes.append(transversals)
            for t_idx in transversals:
                triple_to_boxes[t_idx].append(box_id)
                
    num_boxes = len(boxes)
    
    # 3. Greedy Hitting Set
    box_count = [0] * num_triples
    for t in range(num_triples):
        box_count[t] = len(triple_to_boxes[t])
        
    M = []
    box_hit = [False] * num_boxes
    
    while True:
        if time.time() > deadline:
            break
            
        max_count = -1
        best_t = -1
        for t in range(num_triples):
            if box_count[t] > max_count:
                max_count = box_count[t]
                best_t = t
                
        if max_count <= 0:
            break
            
        M.append(best_t)
        
        for b_idx in triple_to_boxes[best_t]:
            if not box_hit[b_idx]:
                box_hit[b_idx] = True
                for t_idx in boxes[b_idx]:
                    if t_idx != best_t:
                        box_count[t_idx] -= 1
                        
    # 4. Reduction
    # Calculate cover counts
    cover_count = [0] * num_boxes
    for t in M:
        for b_idx in triple_to_boxes[t]:
            cover_count[b_idx] += 1
            
    # Try to remove triples
    # We iterate through M. If we remove one, we update cover_count.
    # To be safe, we can do multiple passes or just one pass.
    # One pass is usually sufficient for greedy+reduce.
    
    M_set = set(M)
    new_M = []
    for t in M:
        # Check if t is essential
        essential = False
        for b_idx in triple_to_boxes[t]:
            if cover_count[b_idx] == 1:
                essential = True
                break
        if not essential:
            # Remove t
            for b_idx in triple_to_boxes[t]:
                cover_count[b_idx] -= 1
        else:
            new_M.append(t)
            
    M = new_M
    
    # 5. Construct result
    M_set = set(M)
    result = []
    for i, t in enumerate(all_triples):
        if i not in M_set:
            result.append(list(t))
            
    return result
