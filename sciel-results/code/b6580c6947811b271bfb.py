   def construct():
       # GF(64) implementation
       def gf64_mul(a, b):
           r = 0
           for i in range(6):
               if b >> i & 1:
                   r ^= a << i
           # Reduce modulo x^6 + x + 1
           # r can be up to 12 bits.
           # If bit 6 is set, r = r ^ (0b1000011 << (i-6))
           for i in range(11, 5, -1):
               if r >> i & 1:
                   r ^= 0b1000011 << (i - 6)
           return r
       
       # Generate log table
       log = {}
       x = 1
       for e in range(63):
           log[x] = e
           x = gf64_mul(x, 2)
           
       n = 62
       colors = []
       for i in range(n):
           for j in range(i + 1, n):
               d = i ^ j
               # d is never 0 since i != j
               c = log[d] % 3
               colors.append(c)
               
       return {'n': n, 'colors': colors}
   