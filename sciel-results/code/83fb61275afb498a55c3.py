import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit

    def is_valid(seq):
        if not seq:
            return True
        if not all(seq[i] < seq[i+1] for i in range(len(seq)-1)):
            return False
        
        sums = set()
        for i in range(len(seq)):
            total = 0
            for j in range(i, len(seq)):
                total += seq[j]
                if total in sums:
                    return False
                sums.add(total)
        return True

    def greedy_build(start_val=1):
        values = []
        sums = set()
        for x in range(start_val, 257):
            tails = []
            total = x
            tails.append(total)
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    def local_search_insert(seq):
        if not seq:
            return seq
        seq_set = set(seq)
        improved = True
        while improved:
            improved = False
            for x in range(1, 257):
                if x in seq_set:
                    continue
                for i in range(len(seq) + 1):
                    if i > 0 and seq[i-1] >= x:
                        continue
                    if i < len(seq) and seq[i] <= x:
                        continue
                    
                    new_seq = seq[:i] + [x] + seq[i:]
                    if is_valid(new_seq):
                        seq = new_seq
                        seq_set.add(x)
                        improved = True
                        break
                if improved:
                    break
                    # Check time?
                    if time.time() > deadline:
                        return seq
        return seq

    best_seq = []
    max_len = 0
    
    # Try a few different greedy starts
    for start_val in [1, 2, 3, 4, 5, 10, 20]:
        if time.time() > deadline:
            break
            
        seq = greedy_build(start_val)
        seq = local_search_insert(seq)
        
        if len(seq) > max_len:
            max_len = len(seq)
            best_seq = seq
            
    return best_seq
