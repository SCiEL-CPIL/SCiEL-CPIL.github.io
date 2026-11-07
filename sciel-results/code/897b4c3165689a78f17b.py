# EVOLVE-BLOCK-START
import time

N = 128

def construct():
    # 1. Precompute forbidden sets
    forbidden_masks = set()
    for c in range(1, N + 1):
        for b in range(1, c):
            if b + c > N:
                break
            for a in range(1, b):
                if a + c > N:
                    break
                # a+b <= N is implied
                f = {a, b, c, a + b, a + c, b + c}
                mask = 0
                for v in f:
                    mask |= (1 << v)
                forbidden_masks.add(mask)
                
    forbidden_list = list(forbidden_masks)
    
    # 2. Precompute propagation data
    # propagate[x] = list of (y, rest_mask)
    # where rest_mask = F \ {x, y} for some forbidden set F containing x and y.
    # If we select x, and rest_mask is subset of current_selected, then y is blocked.
    
    propagate = [[] for _ in range(N + 1)]
    
    for mask in forbidden_list:
        # Get elements in mask
        elements = []
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            elements.append(x)
            m ^= lsb
            
        # For every pair (x, y) in elements, compute rest = mask \ {x, y}
        # We only need to store this for propagation.
        # To avoid duplicates in propagate lists, we can use a set or just allow duplicates 
        # (duplicates just mean checking the same condition twice, which is cheap).
        # However, the list size might grow. 
        # Number of forbidden sets is small (~1000s).
        # Each set has 6 elements, so 15 pairs.
        # Total entries in propagate lists: ~1000 * 15 = 15000. This is fine.
        
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                x = elements[i]
                y = elements[j]
                rest = mask ^ (1 << x) ^ (1 << y)
                propagate[x].append((y, rest))
                propagate[y].append((x, rest))
                
    # 3. Vertex ordering heuristic
    # Sort vertices by number of constraints (degree) descending.
    # We will iterate through available vertices in this static order.
    # To do this efficiently, we can just check available vertices in this order.
    
    degrees = [len(propagate[i]) for i in range(N + 1)]
    vertex_order = list(range(1, N + 1))
    vertex_order.sort(key=lambda x: degrees[x], reverse=True)
    
    # 4. Branch and Bound
    best_size = 0
    best_mask = 0
    
    # Greedy Initialization
    def greedy_init(order):
        S = 0
        available = (1 << (N + 1)) - 2 # Bits 1..N
        for x in order:
            if not (available & (1 << x)):
                continue
            # Check if x can be added
            # Since we are doing greedy, we assume previous choices are fixed.
            # We need to check if x is blocked by S.
            # x is blocked if there exists (y, rest) in propagate[x] such that y in S and rest subset S?
            # No, x is blocked if there exists a forbidden set F containing x such that F \ {x} subset S.
            # This is equivalent to: exists y in S such that (y, rest) in propagate[x] and rest subset S?
            # Not exactly. F \ {x} = {y, ...}. So we need the whole rest.
            # Let's stick to the original check for greedy to be safe, or use the propagate logic.
            # If we use propagate: x is blocked if there is a pair (y, rest) in propagate[x] 
            # such that y is in S AND rest is subset of S?
            # Wait. If F contains x and y, and F \ {x, y} = rest.
            # If y in S and rest subset S, then F \ {x} = {y} U rest subset S. So x is blocked.
            # So yes.
            
            can_add = True
            for y, rest in propagate[x]:
                if (S & (1 << y)) and (rest & ~S) == 0:
                    can_add = False
                    break
            if can_add:
                S |= (1 << x)
                # Update available? Not needed for greedy if we just check S.
                # But we should remove x from available to avoid re-checking? 
                # The loop iterates `order`, so each x is visited once.
        return S

    # Try a few orders
    # 1. Static degree order
    # 2. Reverse degree order
    # 3. Natural order
    # 4. Reverse natural order
    
    orders = [
        vertex_order,
        list(reversed(vertex_order)),
        list(range(1, N + 1)),
        list(range(N, 0, -1))
    ]
    
    for order in orders:
        S = greedy_init(order)
        sz = S.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = S
            
    # B&B Search
    # We use the static vertex_order to pick the next vertex.
    # We iterate through vertex_order, and if the vertex is in available_mask, we branch on it.
    # This avoids the O(N) scan for "best vertex" at each node.
    
    import sys
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    
    def search(selected_mask, available_mask, current_size):
        nonlocal best_size, best_mask
        
        # Time check
        if time.time() - start_time > 58:
            return
            
        # Pruning
        if current_size + available_mask.bit_count() <= best_size:
            return
            
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
            
        # Find the next vertex in vertex_order that is in available_mask
        x = -1
        for v in vertex_order:
            if available_mask & (1 << v):
                x = v
                break
                
        if x == -1:
            return
            
        bit_x = 1 << x
        
        # Branch 1: Exclude x
        search(selected_mask, available_mask & ~bit_x, current_size)
        
        # Branch 2: Include x
        new_selected = selected_mask | bit_x
        new_available = available_mask & ~bit_x
        
        # Propagate: remove y from new_available if blocked by x
        # Check if any (y, rest) in propagate[x] has rest subset of selected_mask
        # Note: We only care about y that are still in new_available
        for y, rest in propagate[x]:
            if (rest & ~selected_mask) == 0:
                # y is blocked
                new_available &= ~(1 << y)
                
        search(new_selected, new_available, current_size + 1)
        
    full_available = (1 << (N + 1)) - 2
    search(0, full_available, 0)
    
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
