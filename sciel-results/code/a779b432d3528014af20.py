<<<<<<< SEARCH
def _known_320_graph():
    """Explicit 4-regular girth-10 graph on 320 vertices (Exoo et al. 2026).
    This is a well-known construction; we encode it compactly.
    Using the (4,10)-cage structure from known tables."""
    # The 320-vertex (4,10)-cage can be constructed using a specific
    # voltage graph / lift construction. Here we use a known explicit
    # edge list encoded via a generator pattern.
    #
    # Construction: Take the 4-regular bipartite graph formed by
    # vertices (i, j) where i in 0..7, j in 0..39 (8*40=320).
    # This is a specific known cage. We encode edges via a rule.
    #
    # Actually, let's use a simpler approach: the (4,10)-cage is known
    # to be constructible. Let me use a verified explicit construction.
    #
    # Using the construction from: "The (4,10)-cage" - we can build
    # it as a 4-fold cover of a smaller base graph.
    #
    # For now, use a systematic construction that is known to work:
    # The 320-vertex graph can be built from the Hoffman-Singleton-like
    # structure or via circulant + lift.
    #
    # Let me use a different approach: construct via the known fact that
    # the (4,10)-cage exists at 320. I'll use a verified edge list
    # generated from the LUW construction if it gives 320, otherwise
    # fall back to a known explicit pattern.
    
    # Actually, let me try a cleaner explicit construction:
    # Vertices: 0..319
    # Use a 4-regular circulant-like structure with girth 10.
    # 
    # Known approach: The (4,10)-cage is a specific graph. Let me
    # construct it using the following verified method:
    # 
    # Use the fact that for k=4, g=10, the cage has 320 vertices.
    # A known construction uses 10 "layers" of 32 vertices each,
    # or 40 layers of 8 vertices.
    #
    # Let me use a simpler verified construction:
    # The graph is the incidence graph of a specific design, or
    # equivalently, we can use a known edge list.
    
    # For reliability, let me generate a valid 4-regular girth-10 graph
    # using a systematic approach that I can verify:
    
    # Method: Start with a 4-regular graph on 320 vertices using
    # a known construction. The simplest verified approach:
    # Use the "cage" construction from algebraic methods.
    
    # Here's a concrete construction:
    # Let V = Z_40 x Z_8 (320 vertices)
    # Each vertex (a, b) connects to:
    #   (a+1, b), (a-1, b), (a, b+1), (a, b-1)  -- this gives girth 4!
    # 
    # That won't work. Need a different connection pattern.
    #
    # Let me use a known valid construction:
    # The (4,10)-cage can be constructed as follows:
    # 
    # Actually, the most reliable approach is to use the LUW construction
    # and verify it gives 320. If not, we need a different method.
    #
    # Let me just return None and let the caller handle it.
    return None
=======
def _known_320_graph():
    """Deprecated: use _cage_4_10 instead."""
    return _cage_4_10()
>>>>>>> REPLACE
