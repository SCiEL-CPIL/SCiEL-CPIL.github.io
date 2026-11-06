import time

def construct():
    deadline = time.time() + 80
    beam = [((), frozenset())]
    best_seq = ()
    W = 5000
    
    while beam and time.time() < deadline:
        next_beam = []
        for seq, prods in beam:
            last = seq[-1] if seq else 1
            for x in range(last + 1, 129):
                new_prods = []
                p = x
                new_prods.append(p)
                for a in reversed(seq):
                    p *= a
                    new_prods.append(p)
                
                if not prods.intersection(new_prods):
                    new_prods_set = prods.union(new_prods)
                    next_beam.append((seq + (x,), new_prods_set))
        
        if not next_beam:
            break
        
        next_beam.sort(key=lambda item: (item[0][-1], sum(item[0])))
        
        if len(next_beam[0][0]) > len(best_seq):
            best_seq = next_beam[0][0]
            
        beam = next_beam[:W]

    return list(best_seq)
