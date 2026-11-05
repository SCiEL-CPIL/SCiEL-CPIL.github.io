import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    n = 34
    
    # Precomputation
    triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triples.append((a, b, c))
    num_triples = len(triples)
    
    triple_index = {}
    for i, t in enumerate(triples):
        triple_index[t] = i
        
    red_sets = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    idxs = [
                        triple_index[(a,b,c)],
                        triple_index[(a,b,d)],
                        triple_index[(a,c,d)],
                        triple_index[(b,c,d)]
                    ]
                    red_sets.append(idxs)
    red_indices = np.array(red_sets, dtype=np.int32)
    
    blue_sets = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    for e in range(d+1, n):
                        idxs = [
                            triple_index[(a,b,c)],
                            triple_index[(a,b,d)],
                            triple_index[(a,b,e)],
                            triple_index[(a,c,d)],
                            triple_index[(a,c,e)],
                            triple_index[(a,d,e)],
                            triple_index[(b,c,d)],
                            triple_index[(b,c,e)],
                            triple_index[(b,d,e)],
                            triple_index[(c,d,e)]
                        ]
                        blue_sets.append(idxs)
    blue_indices = np.array(blue_sets, dtype=np.int32)
    
    # Fallback valid coloring for n=9
    def get_fallback():
        n_fb = 9
        colors_fb = []
        for a in range(n_fb):
            for b in range(a+1, n_fb):
                for c in range(b+1, n_fb):
                    colors_fb.append(0 if (a + b + c) % 3 == 0 else 1)
        return {'n': n_fb, 'colors': colors_fb}

    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
        
        # Local search
        for step in range(3000):
            if time.time() - start_time > time_limit:
                return get_fallback()
                
            red_sums = np.sum(colors[red_indices], axis=1)
            blue_sums = np.sum(colors[blue_indices], axis=1)
            
            red_viol_idx = np.where(red_sums == 0)[0]
            blue_viol_idx = np.where(blue_sums == 10)[0]
            
            num_red_viol = len(red_viol_idx)
            num_blue_viol = len(blue_viol_idx)
            
            if num_red_viol == 0 and num_blue_viol == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            # Fix a random violation
            if num_red_viol > 0 and (num_blue_viol == 0 or random.random() < 0.5):
                set_idx = red_viol_idx[random.randint(0, num_red_viol - 1)]
                t_idx = red_indices[set_idx][random.randint(0, 3)]
                colors[t_idx] = 1
            else:
                # If no blue violations, this branch is skipped.
                # If both exist, 50/50.
                # If only blue exists, we must fix blue.
                if num_blue_viol == 0:
                    # This case is handled by the if above (num_red > 0)
                    # But if num_red == 0 and num_blue == 0, we returned.
                    # So if we are here, and num_red == 0, then num_blue > 0.
                    pass
                set_idx = blue_viol_idx[random.randint(0, num_blue_viol - 1)]
                t_idx = blue_indices[set_idx][random.randint(0, 9)]
                colors[t_idx] = 0
                
            # Random perturbation to escape local minima
            if step % 50 == 0:
                colors[random.randint(0, num_triples - 1)] = random.randint(0, 1)

    return get_fallback()
