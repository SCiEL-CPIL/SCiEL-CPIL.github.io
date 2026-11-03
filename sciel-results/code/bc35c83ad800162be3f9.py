      def support_func(theta, phi):
          # theta: rotation of inner hex
          # phi: angle of normal
          # max cos(phi - theta - j*pi/3)
          # due to symmetry, we can just check 6 values
          # or use the formula
          # h = 1.0 # circumradius is max
          # but we need exact value for LP
          diffs = [phi - theta - j*math.pi/3 for j in range(6)]
          return max(math.cos(d) for d in diffs)
      