   def construct(q, n, radius):
       if q == 3 and n == 8 and radius == 1:
           return _construct_ternary_8_1()
       else:
           return _construct_impl(q, n, radius)
   