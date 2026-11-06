   def construct():
       n = 33
       edges = []
       
       # 8 blocks of K4
       num_blocks = 8
       block_size = 4
       
       for b in range(num_blocks):
           base = b * block_size
           for i in range(block_size):
               for j in range(i + 1, block_size):
                   u = base + i
                   v = base + j
                   edges.append([u, v])
                   
       # Vertex 32
       v_extra = 32
       
       # Connect to all of Block 1 (0,1,2,3)
       for i in range(4):
           edges.append([v_extra, i])
           
       # Connect to first vertex of Block 2..8
       for b in range(1, num_blocks):
           base = b * block_size
           edges.append([v_extra, base])
           
       return {'n': n, 'edges': edges}
   