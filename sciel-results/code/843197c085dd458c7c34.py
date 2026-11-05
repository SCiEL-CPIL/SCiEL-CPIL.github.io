import inspect
import autoresearch_suite.science_tcs_expansion as m
print(inspect.getsource(m.pebble_baseline))
print([n for n in dir(m) if not n.startswith('__')])
