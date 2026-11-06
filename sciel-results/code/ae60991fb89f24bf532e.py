"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    best_A = []
    best_B = []
    best_size = 0
    
    def check_sidon(sets, x):
        """Check if adding x to sets maintains Sidon property (unique pair sums)."""
        for a in sets:
            s = a + x
            # Check if this sum already exists
            for b in sets:
                if b >= a and b + x == s and b != a:
                    return False
                if b + a == s and b > a:
                    return False
        # Check x + x
        s = 2 * x
        for a in sets:
            if a + a == s:
                return False
        return True
    
    def get_diffs(sets):
        """Get all nonzero differences from a set."""
        diffs = set()
        n = len(sets)
        for i in range(n):
            for j in range(i+1, n):
                d = sets[j] - sets[i]
                if d > 0:
                    diffs.add(d)
        return diffs
    
    def run_greedy(order_A, order_B, A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Greedy fill: try to add elements in given order to A and B."""
        nonlocal best_A, best_B, best_size
        
        for x in order_A:
            if x > 256:
                continue
            # Check if x can be added to A
            ok = True
            for a in A:
                s = a + x
                if s in sums_A:
                    ok = False
                    break
            if ok:
                if 2*x in sums_A:
                    ok = False
            if ok:
                # Check differences
                for a in A:
                    d = abs(x - a)
                    if d in diffs_B:
                        ok = False
                        break
            if ok:
                # Add x to A
                for a in A:
                    sums_A.add(a + x)
                sums_A.add(2*x)
                for a in A:
                    d = abs(x - a)
                    diffs_A.add(d)
                A.append(x)
        
        for x in order_B:
            if x > 256:
                continue
            if x in A:
                # Can still add to B if constraints allow
                pass
            ok = True
            for b in B:
                s = b + x
                if s in sums_B:
                    ok = False
                    break
            if ok:
                if 2*x in sums_B:
                    ok = False
            if ok:
                for b in B:
                    d = abs(x - b)
                    if d in diffs_A:
                        ok = False
                        break
            if ok:
                for b in B:
                    sums_B.add(b + x)
                sums_B.add(2*x)
                for b in B:
                    d = abs(x - b)
                    diffs_B.add(d)
                B.append(x)
        
        total = len(A) + len(B)
        if total > best_size:
            best_size = total
            best_A = A[:]
            best_B = B[:]
    
    def run_greedy_interleaved(order, A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Interleaved greedy: alternate between adding to A and B."""
        nonlocal best_A, best_B, best_size
        
        for x in order:
            if x > 256:
                continue
            
            # Try to add to A first
            ok_A = True
            for a in A:
                s = a + x
                if s in sums_A:
                    ok_A = False
                    break
            if ok_A:
                if 2*x in sums_A:
                    ok_A = False
            if ok_A:
                for a in A:
                    d = abs(x - a)
                    if d in diffs_B:
                        ok_A = False
                        break
            
            # Try to add to B
            ok_B = True
            for b in B:
                s = b + x
                if s in sums_B:
                    ok_B = False
                    break
            if ok_B:
                if 2*x in sums_B:
                    ok_B = False
            if ok_B:
                for b in B:
                    d = abs(x - b)
                    if d in diffs_A:
                        ok_B = False
                        break
            
            # Choose which set to add to
            if ok_A and not ok_B:
                for a in A:
                    sums_A.add(a + x)
                sums_A.add(2*x)
                for a in A:
                    d = abs(x - a)
                    diffs_A.add(d)
                A.append(x)
            elif ok_B and not ok_A:
                for b in B:
                    sums_B.add(b + x)
                sums_B.add(2*x)
                for b in B:
                    d = abs(x - b)
                    diffs_B.add(d)
                B.append(x)
            elif ok_A and ok_B:
                # Randomly choose or alternate
                if random.random() < 0.5:
                    for a in A:
                        sums_A.add(a + x)
                    sums_A.add(2*x)
                    for a in A:
                        d = abs(x - a)
                        diffs_A.add(d)
                    A.append(x)
                else:
                    for b in B:
                        sums_B.add(b + x)
                    sums_B.add(2*x)
                    for b in B:
                        d = abs(x - b)
                        diffs_B.add(d)
                    B.append(x)
        
        total = len(A) + len(B)
        if total > best_size:
            best_size = total
            best_A = A[:]
            best_B = B[:]
    
    # Run multiple restarts with different random orderings
    restart_count = 0
    while time.time() - start_time < time_limit:
        restart_count += 1
        
        # Different strategies per restart
        if restart_count % 3 == 1:
            # Random order
            order = list(range(1, 257))
            random.shuffle(order)
        elif restart_count % 3 == 2:
            # Reverse order
            order = list(range(256, 0, -1))
        else:
            # Random with bias toward smaller numbers
            order = list(range(1, 257))
            random.shuffle(order)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        if restart_count % 2 == 1:
            run_greedy_interleaved(order, A, B, sums_A, sums_B, diffs_A, diffs_B)
        else:
            # Split order: first half for A, second half for B
            mid = len(order) // 2
            order_A = order[:mid]
            order_B = order[mid:]
            run_greedy(order_A, order_B, A, B, sums_A, sums_B, diffs_A, diffs_B)
        
        # Check time
        if time.time() - start_time > time_limit - 2:
            break
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
