import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    # Vertices 0..15 correspond to 5-bit strings with even parity
    # We can map integer i (0..15) to a 5-bit string with even parity.
    # There are 16 such strings.
    # List them:
    # Weight 0: 00000 (0)
    # Weight 2: 00011, 00101, 00110, 01001, 01010, 01100, 10001, 10010, 10100, 11000 (10)
    # Weight 4: 00111, 01011, 01101, 01110, 10011, 10101, 10110, 11001, 11010, 11100 (10)
    # Total 1 + 10 + 10 = 21? No.
    # Binomial(5,0)=1, Binomial(5,2)=10, Binomial(5,4)=5.
    # Weight 4: 11110, 11101, 11011, 10111, 01111. (5)
    # Total 1 + 10 + 5 = 16. Correct.
    
    # Map 0..15 to these 16 vectors.
    vectors = []
    # Generate all 5-bit strings with even parity
    for i in range(32):
        # Count bits
        bits = bin(i).count('1')
        if bits % 2 == 0:
            vectors.append(i)
            
    # vectors has 16 elements.
    # We need to map 0..15 to these.
    # Let's just use the index in the list as the vertex ID?
    # The task requires vertices 0..15.
    # We can just create an edge list based on the property of the vectors.
    
    edges = []
    # Adjacency: Hamming distance 4
    for u_idx in range(16):
        for v_idx in range(u_idx + 1, 16):
            u_val = vectors[u_idx]
            v_val = vectors[v_idx]
            # Calculate Hamming distance
            xor = u_val ^ v_val
            dist = bin(xor).count('1')
            if dist == 4:
                edges.append([u_idx, v_idx])
                
    # This graph is the Clebsch graph.
    # It has 40 edges.
    # It is triangle-free and 4-chromatic.
    
    # Verify triangle-free and 4-chromatic?
    # We trust the known properties.
    
    return edges
