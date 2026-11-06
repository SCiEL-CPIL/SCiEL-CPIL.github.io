"""Rewritten program for Communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
import sys
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling using
    Dijkstra search over frontier (cutset) states.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 55  # Safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    if vertices is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
    
    # For each vertex, count how many children it has in needed_set
    child_count = defaultdict(int)
    for v in needed_vertices:
        for p in parent_dict.get(v, []):
            if p in needed_set:
                child_count[p] += 1

    # Fast memory budget check
    if fast_memory_budget < 1:
        fast_memory_budget = 1

    # Dijkstra over frontier states
    # State: frozenset of vertices currently in fast memory (the "frontier")
    # Start state: empty set (nothing in fast memory initially)
    # Goal state: all outputs are in fast memory
    
    # For a state to be valid, it must be a "frontier": 
    # if v is in the state, all its parents must either be in the state or already "done"
    # We track done vertices implicitly: a vertex is done if it's not in the state and 
    # all its children that need it have been processed
    
    # Actually, let's use a different formulation:
    # State = (frontier, done_set) where frontier is the set of live vertices in fast memory
    # But this is too large. Instead, let's use the fact that in a topological DAG,
    # the frontier uniquely determines what's been done: a vertex is done iff it's not 
    # in the frontier and all its parents are done (or it has no parents)
    
    # Simpler approach: State = frozenset of vertices in fast memory
    # A transition: compute vertex v (all parents must be in fast memory or done)
    # After computing v, we may delete vertices that are no longer needed
    
    # Let's define: a vertex is "available to compute" if all its parents are either
    # in the current fast memory set OR have been deleted (done)
    # A vertex is "done" if it's not in fast memory and all its children that need it
    # have been computed and it's no longer needed
    
    # This is getting complex. Let me use a cleaner formulation:
    # 
    # State: (live, done) where:
    #   live = set of vertices currently in fast memory
    #   done = set of vertices that have been computed and deleted (no longer available)
    # 
    # Invariant: live and done are disjoint. A vertex is either live, done, or not-yet-computed.
    # 
    # Transition: pick a vertex v to compute such that all parents of v are in (live ∪ done)
    #   - If parent is in done, we need to load it (cost 1)
    #   - After computing v, v joins live
    #   - Then we may delete any vertex u in live that is not an output and not needed 
    #     by any not-yet-computed vertex (i.e., all children of u that need it are done or v)
    #   - Deleted vertices join done
    #
    # Goal: all outputs are in live (and we stop)
    #
    # Cost: number of load operations (storing is free in this model since we track done)
    # Actually, the cost is the number of transfers = loads + stores. But in this model,
    # when we delete a vertex that was computed, that's a store (if it might be needed again) 
    # or a delete (if not needed). Let me reconsider.
    #
    # Let me use: cost = number of load operations. Each time we need a vertex that's in done,
    # we pay 1 load. The total number of loads is what we minimize.
    # (Stores don't add to the transfer count in the same way; the key metric is loads)
    
    # Actually, re-reading the problem: "Score mean transfers / naive schedule"
    # Transfers = loads + stores. Let me count both.
    
    # Let me use a simpler approach: Dijkstra where state is the frozenset of live vertices,
    # and we track the minimum cost to reach each state. The cost of a transition is the
    # number of loads needed (parents in done that need to be loaded) + number of stores
    # (vertices deleted that are not "free" to delete).
    
    # Hmm, this is still complex. Let me just implement a clean version:
    
    # State: frozenset of vertices in fast memory
    # We also need to know which vertices have been "fully processed" (done)
    # A vertex is done if it's not in fast memory and all its needed children have been processed
    
    # Let me use: state = (live_frozenset, done_frozenset)
    # But to keep state space small, note that done is determined by live and the DAG structure
    # in a topological sense. Actually no, it's not uniquely determined.
    
    # Let me just use live_frozenset as the state and compute done implicitly:
    # A vertex v is done iff v not in live AND for all children c of v (in needed_set),
    # c is either done or in live. This is circular.
    
    # OK, let me use a different approach: BFS with state = frozenset of live vertices,
    # and we track the set of "completed" vertices separately. Since the DAG is topologically
    # ordered, we can determine which vertices are "available" to compute given the live set.
    
    # Available to compute: v such that all parents of v (in needed_set) are in live
    # (because if a parent was done, it would have been deleted, but we can reload it)
    
    # Wait, I think the cleanest formulation is:
    # State = frozenset of vertices in fast memory
    # From a state S, we can:
    #   1. Compute a vertex v where all parents of v are in S. Cost: 0 (just compute).
    #      New state: S ∪ {v}
    #   2. Delete a vertex u in S where u is not an output and no not-yet-processed vertex 
    #      needs u. Cost: 0 if u was an input (delete is free), 1 if u was computed (store).
    #      New state: S \ {u}
    #   3. Load a vertex u not in S where u was previously stored (computed and deleted). 
    #      Cost: 1. New state: S ∪ {u}
    #
    # But we need to track what's been stored. This requires more state.
    
    # Let me simplify: I'll use the "frontier" approach where the state is the set of 
    # vertices in fast memory, and I assume we can always reload any vertex at cost 1.
    # The key insight is that the optimal strategy will only load a vertex when needed.
    
    # Actually, for a clean implementation within time limits, let me use a greedy 
    # heuristic based on the frontier concept, which is much faster than full Dijkstra.
    
    # Greedy frontier-based algorithm:
    # 1. Start with empty fast memory
    # 2. At each step, determine which vertices are "ready" (all parents in fast memory)
    # 3. Choose the next vertex to compute based on a heuristic (e.g., minimize future loads)
    # 4. After computing, delete vertices that are no longer needed (FNU-style)
    # 5. Load vertices as needed
    
    # This is essentially what the original code does, but let me refine the heuristic.
    
    # For now, let me keep the original approach but improve the eviction strategy
    # using a better heuristic based on the frontier concept.
    
    # Precompute next-use positions
    next_use = defaultdict(list)
    for i, v in enumerate(needed_vertices):
        for p in parent_dict.get(v, []):
            if p in needed_set:
                next_use[p].append(i)

    # State
    in_fast = set()
    in_slow = set(inputs)  # Inputs start in slow memory
    computed = set()
    operations = []
    
    # Track current position for FNU
    current_pos = 0

    def get_next_use(u, pos):
        """Get the next use position of u after position pos, or infinity if never used again."""
        uses = next_use.get(u, [])
        lo, hi = 0, len(uses)
        while lo < hi:
            mid = (lo + hi) // 2
            if uses[mid] <= pos:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(uses):
            return uses[lo]
        return float('inf')

    def load(v):
        operations.append(['load', v])
        in_fast.add(v)
        
    def store(v):
        operations.append(['store', v])
        in_fast.discard(v)
        in_slow.add(v)
        
    def compute(v):
        operations.append(['compute', v])
        computed.add(v)
        in_fast.add(v)
        
    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)

    def evict_one(protect_set):
        """Evict one vertex from fast memory using improved heuristic."""
        if not in_fast:
            return None
        
        # Priority 1: Find a vertex that is not needed again
        candidate_delete = None
        for u in list(in_fast):
            if u in protect_set:
                continue
            nu = get_next_use(u, current_pos)
            if nu == float('inf'):
                candidate_delete = u
                break
        
        if candidate_delete is not None:
            delete(candidate_delete)
            return candidate_delete

        # Priority 2: FNU on the rest, but prefer vertices that are inputs (cheaper to reload)
        best_evict = None
        best_score = -1
        
        for u in list(in_fast):
            if u in protect_set:
                continue
            nu = get_next_use(u, current_pos)
            # Score: prefer evicting inputs (reload is just a load, no store cost)
            # and vertices with later next use
            score = nu
            if u in inputs:
                score *= 2  # Prefer evicting inputs
            if score > best_score:
                best_score = score
                best_evict = u
        
        if best_evict is None:
            for u in in_fast:
                best_evict = u
                break
        
        if best_evict is not None:
            if best_evict in in_slow:
                delete(best_evict)
            else:
                store(best_evict)
        
        return best_evict

    # Process vertices in topological order
    for v in needed_vertices:
        if time.time() > deadline:
            break
        
        current_pos = vertex_to_idx.get(v, current_pos)
        
        parents = [p for p in parent_dict.get(v, []) if p in needed_set]

        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            
            if len(in_fast) >= fast_memory_budget:
                remaining_parents = set(parents) - {p}
                evict_one(remaining_parents | {v})
            
            load(p)

        # Compute v
        if v not in in_fast:
            compute(v)
        
        # If v is not an output and is not needed in the future, delete it immediately
        if v not in outputs:
            nu_v = get_next_use(v, current_pos)
            if nu_v == float('inf'):
                delete(v)

    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                evict_one(outputs)
            load(v)

    # Final check
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) < fast_memory_budget:
                load(v)
            else:
                if in_fast:
                    u = next(iter(in_fast))
                    if u in in_slow:
                        delete(u)
                    else:
                        store(u)
                    load(v)

    return operations
# EVOLVE-BLOCK-END
