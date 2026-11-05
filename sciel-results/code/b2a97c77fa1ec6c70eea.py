import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_seq = []
    best_len = 0
    
    BEAM_WIDTH = 400
    MAX_CANDIDATES = 100
    
    # Beams: (seq, product_set, suffix_prods)
    # suffix_prods[i] is the product of the last i+1 elements of seq.
    # e.g. seq=[2,3,5], suffix_prods=[5, 15, 30]
    
    beams = [([], set(), [])]
    
    while time.time() < deadline:
        new_beams = []
        
        for seq, prods, suffix_prods in beams:
            if not seq:
                last_val = 1
            else:
                last_val = seq[-1]
            
            end = min(last_val + MAX_CANDIDATES + 1, 129)
            
            for c in range(last_val + 1, end):
                collision = False
                if c in prods:
                    collision = True
                else:
                    for s in suffix_prods:
                        if c * s in prods:
                            collision = True
                            break
                
                if not collision:
                    new_seq = seq + [c]
                    new_len = len(new_seq)
                    
                    new_suffix_prods = [c]
                    for s in suffix_prods:
                        new_suffix_prods.append(c * s)
                    
                    new_prods = prods.copy()
                    new_prods.add(c)
                    for s in suffix_prods:
                        new_prods.add(c * s)
                    
                    new_beams.append((new_seq, new_prods, new_suffix_prods))
                    
                    if new_len > best_len:
                        best_len = new_len
                        best_seq = new_seq
        
        if not new_beams:
            break
        
        def beam_sort_key(b):
            seq, prods, _ = b
            return (-len(seq), seq[-1] if seq else 0)
        
        new_beams.sort(key=beam_sort_key)
        beams = new_beams[:BEAM_WIDTH]
        
        if best_len >= 35:
            BEAM_WIDTH = 200
            MAX_CANDIDATES = 60

    return best_seq
