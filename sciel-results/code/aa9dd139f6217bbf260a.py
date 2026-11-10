import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 34
    
    # Precompute indices for fast lookup
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4_list = list(combinations(range(n), 4))
    num_k4 = len(k4_list)
    k5_list = list(combinations(range(n), 5))
    num_k5 = len(k5_list)
    
    k4_data = []
    triple_to_k4s = [[] for _ in range(num_triples)]
    for i, k4 in enumerate(k4_list):
        ts = []
        for j in range(4):
            t = tuple(sorted(k4[:j] + k4[j+1:]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k4s[idx].append(i)
        k4_data.append(ts)
        
    k5_data = []
    triple_to_k5s = [[] for _ in range(num_triples)]
    for i, k5 in enumerate(k5_list):
        ts = []
        for combo in combinations(range(5), 3):
            t = tuple(sorted([k5[j] for j in combo]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k5s[idx].append(i)
        k5_data.append(ts)

    k4_arr = np.array(k4_data, dtype=np.int32)
    k5_arr = np.array(k5_data, dtype=np.int32)
    
    # Helper class to manage state efficiently
    class State:
        def __init__(self, colors):
            self.colors = colors
            self.counts_k4 = np.sum(colors[k4_arr], axis=1)
            self.counts_k5 = np.sum(colors[k5_arr], axis=1)
            # Red K4: all 4 triples are 0 (sum=0)
            # Blue K5: all 10 triples are 1 (sum=10)
            self.violations = np.sum(self.counts_k4 == 0) + np.sum(self.counts_k5 == 10)
            
        def flip(self, idx):
            old_val = self.colors[idx]
            new_val = 1 - old_val
            delta = new_val - old_val # +1 if 0->1, -1 if 1->0
            
            # Update K4s
            for k4_idx in triple_to_k4s[idx]:
                old_count = self.counts_k4[k4_idx]
                new_count = old_count + delta
                self.counts_k4[k4_idx] = new_count
                old_viol = 1 if old_count == 0 else 0
                new_viol = 1 if new_count == 0 else 0
                self.violations += (new_viol - old_viol)
                
            # Update K5s
            for k5_idx in triple_to_k5s[idx]:
                old_count = self.counts_k5[k5_idx]
                new_count = old_count + delta
                self.counts_k5[k5_idx] = new_count
                old_viol = 1 if old_count == 10 else 0
                new_viol = 1 if new_count == 10 else 0
                self.violations += (new_viol - old_viol)
            
            self.colors[idx] = new_val
            
        def get_random_violation(self):
            # Returns (type, idx_in_list, list_of_edges)
            # type 0: Red K4, type 1: Blue K5
            if self.violations == 0:
                return None
            
            # Randomly pick a violation type weighted by count
            red_k4_count = np.sum(self.counts_k4 == 0)
            blue_k5_count = np.sum(self.counts_k5 == 10)
            
            if red_k4_count > 0 and (blue_k5_count == 0 or random.random() < red_k4_count / (red_k4_count + blue_k5_count)):
                # Pick random red K4
                violating_k4s = np.where(self.counts_k4 == 0)[0]
                k4_idx = violating_k4s[random.randint(0, len(violating_k4s) - 1)]
                return (0, k4_idx, k4_data[k4_idx])
            else:
                # Pick random blue K5
                violating_k5s = np.where(self.counts_k5 == 10)[0]
                k5_idx = violating_k5s[random.randint(0, len(violating_k5s) - 1)]
                return (1, k5_idx, k5_data[k5_idx])

    best_violations = float('inf')
    best_colors = None
    
    # Min-Conflicts Local Search
    max_restarts = 1000
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        state = State(colors)
        
        # Local Search
        max_steps = 5000
        no_improve = 0
        best_state_violations = state.violations
        best_state_colors = state.colors.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            if state.violations == 0:
                best_violations = 0
                best_colors = state.colors.copy()
                break
                
            if state.violations < best_state_violations:
                best_state_violations = state.violations
                best_state_colors = state.colors.copy()
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve > 1000:
                break # Stuck, restart
                
            # Find a violation
            viol_info = state.get_random_violation()
            if viol_info is None:
                break
                
            _, _, edges = viol_info
            
            # Try flipping each edge in the violation to see if it improves
            best_delta = float('inf')
            best_edge = None
            
            # We can sample edges if the list is long, but K4 has 4, K5 has 10. Check all.
            for edge_idx in edges:
                # Save current violation count
                old_viol = state.violations
                
                # Flip
                state.flip(edge_idx)
                
                # Calculate delta
                delta = state.violations - old_viol
                
                # Revert
                state.flip(edge_idx)
                
                if delta < best_delta:
                    best_delta = delta
                    best_edge = edge_idx
                    
            # Apply best edge
            if best_edge is not None:
                state.flip(best_edge)
                
                # If best_delta was positive (worsening), we might want to randomize
                # to escape local minima. 
                # If all flips worsen, we pick a random one from the violation
                # or a global random flip.
                if best_delta > 0:
                    # Escape strategy: pick random edge from violation
                    escape_edge = random.choice(edges)
                    # We already flipped best_edge, so we need to flip it back
                    # and flip the escape edge.
                    # Actually, let's just re-evaluate.
                    # The current state has best_edge flipped.
                    # If best_delta > 0, it's worse than before.
                    # Let's just do a global random flip to break symmetry.
                    state.flip(best_edge) # Revert best_edge
                    global_edge = random.randint(0, num_triples - 1)
                    state.flip(global_edge)
            else:
                # Should not happen if violation exists
                break
                
        # End local search
        if best_state_violations < best_violations:
            best_violations = best_state_violations
            best_colors = best_state_colors.copy()
            
        if best_violations == 0:
            break

    if best_violations == 0:
        return {'n': 34, 'colors': best_colors.tolist()}
    else:
        # Fallback: A guaranteed valid coloring for n=4
        # Vertices 0,1,2,3
        # Triples: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
        # Set (0,1,2) to Blue (1), others Red (0)
        # This avoids Red K4 (since one is Blue) and n<5 avoids Blue K5
        n_fallback = 4
        colors_fallback = [1, 0, 0, 0] # Lexicographic order: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
        return {'n': n_fallback, 'colors': colors_fallback}
