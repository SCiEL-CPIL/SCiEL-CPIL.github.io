import time

def construct():
    start_time = time.time()
    beam_width = 20
    beam = [([], set(), [0])]
    
    for step in range(250):
        if time.time() - start_time > 55:
            break
        candidates = []
        for seq, sums, suffix in beam:
            n = len(seq)
            for x in range(1, 257):
                ok = True
                for i in range(n+1):
                    if x + suffix[i] in sums:
                        ok = False
                        break
                if ok:
                    new_suffix = [0] * (n+2)
                    new_suffix[0] = 0
                    for i in range(n+1):
                        new_suffix[i+1] = x + suffix[i]
                    new_sums = sums.union(new_suffix[1:])
                    new_seq = seq + [x]
                    candidates.append((x, new_seq, new_sums, new_suffix))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda c: c[0])
        beam = [c[1:] for c in candidates[:beam_width]]
    
    best = max(beam, key=lambda x: len(x[0]))
    return best[0]
