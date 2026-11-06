import time
import random
import math
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

# Precompute edge index mapping
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all edges as (u, v) tuples for easier manipulation
EDGES = []
for i in range(N):
    for j in range(i + 1, N):
        EDGES.append((i, j))

# Map edge tuple to index
EDGE_TO_IDX = {e: idx for idx, e in enumerate(EDGES)}

# Precompute all 4-cycles (as sets of 4 edge indices)
# A 4-cycle on vertices a,b,c,d can be formed in 3 ways:
# 1. (a,b), (b,c), (c,d), (d,a)
# 2. (a,b), (b,d), (d,c), (c,a)
# 3. (a,c), (c,b), (b,d), (d,a)
CYCLES = []
CYCLE_SETS = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                # Cycle 1
                e1 = edge_idx(a, b)
                e2 = edge_idx(b, c)
                e3 = edge_idx(c, d)
                e4 = edge_idx(d, a)
                CYCLES.append((e1, e2, e3, e4))
                CYCLE_SETS.append(frozenset([e1, e2, e3, e4]))
                
                # Cycle 2
                e1 = edge_idx(a, b)
                e2 = edge_idx(b, d)
                e3 = edge_idx(d, c)
                e4 = edge_idx(c, a)
                CYCLES.append((e1, e2, e3, e4))
                CYCLE_SETS.append(frozenset([e1, e2, e3, e4]))
                
                # Cycle 3
                e1 = edge_idx(a, c)
                e2 = edge_idx(c, b)
                e3 = edge_idx(b, d)
                e4 = edge_idx(d, a)
                CYCLES.append((e1, e2, e3, e4))
                CYCLE_SETS.append(frozenset([e1, e2, e3, e4]))

# For each edge, list the cycle indices it belongs to
EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(CYCLES):
    for e in cyc:
        EDGE_CYCLES[e].append(ci)

def is_c4_free(edge_set):
    """Check if a set of edges contains any C4."""
    if not edge_set:
        return True
    # For each edge in the set, check if it completes a C4 with other edges in the set
    # Optimization: just check all cycles
    for cyc_set in CYCLE_SETS:
        if cyc_set.issubset(edge_set):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin
    
    # Strategy: Greedy Set Packing
    # We want to partition edges into color classes C_1, ..., C_k
    # Constraint: For any i != j, C_i U C_j is C4-free.
    # This ensures no rainbow C4 (which would need 4 distinct colors).
    # Actually, the constraint "No four-cycle may have four distinct edge colors"
    # is equivalent to: For any C4, at least two edges share a color.
    # If we ensure that for any two color classes C_i, C_j, their union is C4-free,
    # then a C4 cannot have edges from 3 or more distinct classes.
    # If a C4 has edges from exactly 2 classes, say C_i and C_j, then since C_i U C_j is C4-free,
    # that C4 cannot exist.
    # Wait, if C_i U C_j is C4-free, it means there is NO C4 formed entirely by edges in C_i U C_j.
    # So a C4 cannot have edges ONLY from C_i and C_j.
    # It could have edges from C_i, C_j, and C_k.
    # Let's re-read the condition: "No four-cycle may have four distinct edge colors."
    # This allows 1, 2, or 3 distinct colors on a C4.
    # It forbids 4 distinct colors.
    
    # The proposed strategy in the prompt: "union of any two subgraphs G_i U G_j is C4-free"
    # If this holds, then any C4 must use edges from at least 3 distinct color classes?
    # No. If a C4 uses edges from C1 and C2 only, then those edges are in C1 U C2.
    # If C1 U C2 is C4-free, then that C4 cannot exist.
    # So any existing C4 must use edges from at least 3 color classes.
    # But a C4 has 4 edges. By Pigeonhole, if it uses 3 colors, one color appears at least twice.
    # So it doesn't have 4 distinct colors.
    # Thus, the condition "For all i != j, C_i U C_j is C4-free" is a SUFFICIENT condition
    # for the constraint "No C4 has 4 distinct colors".
    
    # Goal: Maximize k (number of colors).
    
    # Algorithm:
    # 1. Start with all edges uncolored.
    # 2. Iteratively select a maximal set of edges S_k such that:
    #    a) S_k is disjoint from all previously selected S_1...S_{k-1}.
    #    b) For all i < k, S_k U S_i is C4-free.
    #    c) S_k is as large as possible (or just maximal) to allow more colors?
    #    Actually, we want to maximize k. So we want each S_k to be as SMALL as possible?
    #    No, we want to fit as many sets as possible.
    #    If we make S_k small, we might fit more sets.
    #    But S_k must be a valid color class.
    #    Let's try to make S_k a matching? Or just a single edge?
    #    If S_k is a single edge, condition (b) is: {e} U S_i is C4-free.
    #    Since S_i is C4-free (implied by condition b for i=j? No, condition is for i!=j).
    #    Wait, if S_i is not C4-free, then a C4 inside S_i has 1 color, which is allowed.
    #    But if S_i U S_j is C4-free, then S_i itself must be C4-free?
    #    If S_i contains a C4, then S_i U S_j contains that C4. So S_i U S_j is NOT C4-free.
    #    Therefore, the condition implies that EACH S_i is C4-free.
    #    
    #    So the problem reduces to: Decompose E(K_16) into k subgraphs G_1...G_k
    #    such that each G_i is C4-free, and for all i!=j, G_i U G_j is C4-free.
    #    Maximize k.
    
    # Greedy Heuristic:
    # Try to assign each edge to a new color class if possible, or an existing one?
    # To maximize k, we want to create as many NEW classes as possible.
    # So we should try to assign each edge to a NEW class if it doesn't violate constraints.
    # If it does, assign to an existing class.
    
    # Let's try a randomized greedy approach with restarts.
    
    best_k = 0
    best_assignment = None
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        
        # Order of edges
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        # current_classes: list of sets of edge indices
        current_classes = []
        
        # For fast checking, maintain for each class the set of edges
        # And for each pair of classes, check C4-freeness of union?
        # That's expensive.
        
        # Alternative:
        # When adding edge e to class C_new:
        # Check if C_new U C_old is C4-free for all existing C_old.
        
        # To speed up, we can precompute for each edge e, the list of C4s containing e.
        # A union C_new U C_old contains a C4 iff there is a C4 where all 4 edges are in C_new U C_old.
        # Since C_new is built incrementally, this is tricky.
        
        # Let's simplify:
        # We will build classes one by one.
        # For the next class, we pick a maximal set of remaining edges such that:
        # 1. It is C4-free.
        # 2. Union with any existing class is C4-free.
        
        # This is still hard to check dynamically.
        
        # Let's try a simpler heuristic:
        # Assign edges to classes in a random order.
        # For each edge e:
        #   Try to put it in a new class (if possible).
        #   If not, try to put it in an existing class.
        #   If not, fail (or restart).
        
        # "Possible" means:
        # If new class: {e} U C_i is C4-free for all existing C_i.
        # Since {e} is C4-free, we just need to check if adding e to C_i creates a C4 in the union.
        # This happens if there is a C4 containing e and 3 other edges in C_i.
        
        # Precompute: For each edge e, and each other edge f, do they share a C4?
        # No, we need 3 edges in C_i that with e form a C4.
        
        # Let's precompute for each edge e, the set of "conflicts".
        # A conflict for e with respect to a set S is a C4 containing e and 3 edges from S.
        
        # This is getting complex. Let's use the property that N=16 is small.
        # Number of C4s is ~ 16*15*14*13/24 * 3?
        # C(16,4) * 3 = 1820 * 3 = 5460 C4s.
        
        # We can maintain for each class C_i a bitmask of edges? No, 120 bits.
        # We can just check validity by iterating over C4s.
        
        # Let's try the "New Class First" greedy.
        
        valid = True
        used_edges = set()
        
        for e in edge_order:
            if e in used_edges:
                continue
                
            # Try to create a new class with just {e}
            # Check if {e} U C_i is C4-free for all existing C_i
            can_new = True
            for C_i in current_classes:
                # Check if C_i U {e} is C4-free
                # This is equivalent to: No C4 in C_i U {e}
                # Since C_i is C4-free (by induction), the only new C4s would involve e.
                # So check if there is a C4 containing e and 3 edges from C_i.
                for ci in EDGE_CYCLES[e]:
                    cyc = CYCLES[ci]
                    # e is in cyc. Check if other 3 edges are in C_i.
                    others = [x for x in cyc if x != e]
                    if all(x in C_i for x in others):
                        can_new = False
                        break
                if not can_new:
                    break
            
            if can_new:
                current_classes.append({e})
                used_edges.add(e)
                continue
            
            # Try to add e to an existing class
            added = False
            # Sort existing classes by size? Or random order?
            # Random order to avoid bias
            class_indices = list(range(len(current_classes)))
            random.shuffle(class_indices)
            
            for ci_idx in class_indices:
                C_i = current_classes[ci_idx]
                # Check if C_i U {e} is C4-free
                # Again, check if there is a C4 containing e and 3 edges from C_i.
                # Wait, C_i is C4-free. So any C4 in C_i U {e} must use e.
                valid_add = True
                for cyc_idx in EDGE_CYCLES[e]:
                    cyc = CYCLES[cyc_idx]
                    others = [x for x in cyc if x != e]
                    if all(x in C_i for x in others):
                        valid_add = False
                        break
                if valid_add:
                    C_i.add(e)
                    used_edges.add(e)
                    added = True
                    break
            
            if not added:
                valid = False
                break
        
        if valid:
            k = len(current_classes)
            if k > best_k:
                best_k = k
                # Create final assignment
                assignment = [0] * NUM_EDGES
                color_id = 0
                for C_i in current_classes:
                    for e in C_i:
                        assignment[e] = color_id
                    color_id += 1
                best_assignment = assignment[:]
                
    # If we didn't find any valid assignment (unlikely), return trivial
    if best_assignment is None:
        return [0] * NUM_EDGES
        
    return best_assignment

# Helper for verification if needed, but not in construct
def verify(assignment):
    for cyc in CYCLES:
        colors = [assignment[e] for e in cyc]
        if len(set(colors)) == 4:
            return False
    return True

# For testing
if __name__ == "__main__":
    res = construct()
    print("Max colors:", len(set(res)))
    print("Valid:", verify(res))
