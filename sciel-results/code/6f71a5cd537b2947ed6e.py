import time
import random
import numpy as np

BASE = 51975

def get_divisors(n):
    divs = []
    for i in range(2, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs

DIVISORS = get_divisors(BASE)
DIVISORS.sort()
NUM_DIVS = len(DIVISORS)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_score = BASE
    best_classes = None
    
    def greedy_init(order):
        """Greedy: process moduli in given order, pick best residue for each."""
        covered = bytearray(BASE)
        classes = []
        for m in order:
            # Find residue r that covers most uncovered
            best_r = 0
            best_count = -1
            # Use bytearray slicing for speed
            for r in range(m):
                cnt = 0
                for i in range(r, BASE, m):
                    if covered[i] == 0:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            if best_count > 0:
                for i in range(best_r, BASE, m):
                    covered[i] = 1
                classes.append([m, best_r])
        return classes, covered
    
    def greedy_init_fast(order):
        """Faster greedy using numpy for residue selection but bytearray for tracking."""
        covered = np.zeros(BASE, dtype=np.uint8)
        classes = []
        for m in order:
            # Reshape to find best residue
            try:
                c_reshaped = covered.reshape(-1, m)
                uncovered_counts = m * (BASE // m) - np.sum(c_reshaped, axis=0)
                # Actually: total in class = BASE // m, covered = sum, uncovered = total - covered
                total_in_class = BASE // m
                uncovered_counts = total_in_class - np.sum(c_reshaped, axis=0)
                best_r = int(np.argmax(uncovered_counts))
                best_count = uncovered_counts[best_r]
            except:
                best_r = 0
                best_count = 0
            
            if best_count > 0:
                covered[best_r::m] = 1
                classes.append([m, best_r])
        return classes, covered
    
    def local_search(classes, covered_np, max_time=30.0):
        """Local search: reassign residues and swap moduli."""
        start = time.time()
        current_score = int(np.sum(covered_np == 0))
        
        # Build used_moduli set
        used_moduli = set(m for m, r in classes)
        
        while time.time() - start < max_time:
            improved = False
            
            # Pass 1: Try reassigning residue for each existing class
            for idx in range(len(classes)):
                m, r_old = classes[idx]
                
                # Compute marginal: how many become uncovered if we remove this class
                z_created = int(np.sum(covered_np[r_old::m] == 1))
                
                # Temporarily remove
                covered_np[r_old::m] -= 1
                
                # Find best new residue
                try:
                    c_reshaped = covered_np.reshape(-1, m)
                    gains = np.sum(c_reshaped == 0, axis=0)
                    best_r_new = int(np.argmax(gains))
                    z_removed = int(gains[best_r_new])
                except:
                    best_r_new = r_old
                    z_removed = 0
                
                delta = z_created - z_removed
                
                # Revert
                covered_np[r_old::m] += 1
                
                if delta < 0:
                    # Apply improvement
                    covered_np[r_old::m] -= 1
                    covered_np[best_r_new::m] += 1
                    classes[idx] = [m, best_r_new]
                    current_score += delta
                    improved = True
                    if current_score == 0:
                        return classes, current_score
            
            # Pass 2: Try swapping moduli
            unused = [m for m in DIVISORS if m not in used_moduli]
            if unused:
                for idx in range(len(classes)):
                    m_old, r_old = classes[idx]
                    
                    for m_new in unused[:10]:  # Limit swaps tried per iteration
                        # Remove old
                        z_created_old = int(np.sum(covered_np[r_old::m_old] == 1))
                        covered_np[r_old::m_old] -= 1
                        
                        # Find best r for m_new
                        try:
                            c_reshaped = covered_np.reshape(-1, m_new)
                            gains = np.sum(c_reshaped == 0, axis=0)
                            best_r_new = int(np.argmax(gains))
                            z_removed_new = int(gains[best_r_new])
                        except:
                            best_r_new = 0
                            z_removed_new = 0
                        
                        delta = z_created_old - z_removed_new
                        
                        # Revert
                        covered_np[r_old::m_old] += 1
                        
                        if delta < 0:
                            # Apply swap
                            covered_np[r_old::m_old] -= 1
                            covered_np[best_r_new::m_new] += 1
                            classes.pop(idx)
                            classes.append([m_new, best_r_new])
                            used_moduli.remove(m_old)
                            used_moduli.add(m_new)
                            current_score += delta
                            improved = True
                            if current_score == 0:
                                return classes, current_score
                            break  # Restart pass after a swap
            
            if not improved:
                break
        
        return classes, current_score
    
    # Multi-restart local search
    num_restarts = 8
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Different ordering for each restart
        if restart == 0:
            order = DIVISORS[:]  # Ascending
        elif restart == 1:
            order = DIVISORS[::-1]  # Descending
        else:
            order = DIVISORS[:]
            random.shuffle(order)
        
        classes, covered_np = greedy_init_fast(order)
        current_score = int(np.sum(covered_np == 0))
        
        if current_score < best_score:
            best_score = current_score
            best_classes = classes[:]
        
        if best_score == 0:
            break
        
        # Local search
        time_left = deadline - 5 - time.time()
        if time_left > 2:
            classes, current_score = local_search(classes, covered_np, max_time=time_left)
            if current_score < best_score:
                best_score = current_score
                best_classes = classes[:]
            
            if best_score == 0:
                break
    
    if best_classes is None:
        # Fallback: simple greedy
        best_classes, _ = greedy_init_fast(DIVISORS)
    
    return best_classes

random.seed(42)
